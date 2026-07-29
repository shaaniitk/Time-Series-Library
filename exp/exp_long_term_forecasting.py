from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, combine_primary_and_aux_loss, get_auxiliary_loss, visual
import json
from utils.metrics import metric, quantile_metric
from utils.losses import QuantileLoss
from utils.tft_schema import (
    inverse_transform_selected,
    is_tft_model,
    resolve_target_positions,
    select_tft_truth,
)
from utils.tft_config import apply_tft_profile
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        if is_tft_model(args):
            args = apply_tft_profile(args)
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._tft_target_positions = resolve_target_positions(args) if is_tft_model(args) else None
        self._tft_output_mode = str(getattr(args, 'tft_output_mode', 'joint' if getattr(args, 'tft_use_quantile_head', False) else 'point')).lower()
        self._point_loss_coeff = float(getattr(args, 'tft_point_loss_coeff', 1.0))
        self._quantile_loss_coeff = float(getattr(args, 'tft_quantile_loss_coeff', 1.0))
        quantiles = getattr(args, 'tft_output_quantiles', None)
        self._joint_quantile_criterion = QuantileLoss(quantiles) if is_tft_model(args) and self._tft_output_mode in {'quantile', 'joint'} else None
        self._quantile_levels = tuple(self._joint_quantile_criterion.quantiles) if self._joint_quantile_criterion is not None else None
        self._validate_tft_objective_config()

    def _validate_tft_objective_config(self):
        if self._tft_target_positions is None:
            return
        if self._tft_output_mode not in {'point', 'quantile', 'joint'}:
            raise ValueError("tft_output_mode must be one of: point, quantile, joint.")
        if self._tft_output_mode == 'joint':
            if self._point_loss_coeff <= 0.0 or self._quantile_loss_coeff <= 0.0:
                raise ValueError("tft_output_mode=joint requires positive tft_point_loss_coeff and tft_quantile_loss_coeff.")
        if self._tft_output_mode in {'quantile', 'joint'} and self._joint_quantile_criterion is None:
            raise ValueError(f"tft_output_mode={self._tft_output_mode} requires tft_output_quantiles.")

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if str(getattr(self.args, 'loss', 'MSE')).upper() == 'QUANTILE':
            quantiles = getattr(self.args, 'tft_output_quantiles', None)
            if not getattr(self.args, 'tft_use_quantile_head', False):
                raise ValueError("loss=Quantile requires tft_use_quantile_head=True.")
            return QuantileLoss(quantiles)
        if self._tft_output_mode == 'quantile':
            raise ValueError("tft_output_mode=quantile requires loss=Quantile.")
        criterion = nn.MSELoss()
        return criterion

    def _get_aux_loss_coeff(self):
        return float(getattr(self.args, 'tft_moe_aux_loss_coeff', 0.0))

    @staticmethod
    def _cv_squared(x):
        eps = 1e-10
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self._tft_target_positions is None:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_auxiliary=True)

    def _extract_outputs_and_aux(self, model_output):
        if self._tft_target_positions is None:
            return model_output, None, get_auxiliary_loss(self.model)

        if model_output.moe_importance_sum is not None:
            importance_sum = model_output.moe_importance_sum
            load_sum = getattr(model_output, 'moe_load_sum', None)
            if importance_sum.ndim == 1:
                importance_sum = importance_sum.unsqueeze(0)
            importance_sum = importance_sum.sum(dim=0)
            aux_loss = self._cv_squared(importance_sum)
            if load_sum is not None:
                if load_sum.ndim == 1:
                    load_sum = load_sum.unsqueeze(0)
                load_sum = load_sum.sum(dim=0)
                aux_loss = aux_loss + self._cv_squared(load_sum)
        else:
            aux_loss = None
        return model_output.point_full, model_output.quantile_forecast, aux_loss

    def _compute_supervised_loss(self, outputs, true, criterion, quantile_outputs):
        if self._tft_target_positions is None:
            if isinstance(criterion, QuantileLoss):
                if quantile_outputs is None:
                    raise RuntimeError("Quantile loss selected but model did not populate last_quantile_predictions.")
                return criterion(quantile_outputs, true)
            return criterion(outputs, true)

        if self._tft_output_mode == 'point':
            return criterion(outputs, true)

        if self._tft_output_mode == 'quantile':
            if quantile_outputs is None:
                raise RuntimeError("tft_output_mode=quantile requires model quantile outputs.")
            return criterion(quantile_outputs, true)

        if self._point_loss_coeff <= 0.0 or self._quantile_loss_coeff <= 0.0:
            raise ValueError("tft_output_mode=joint requires positive tft_point_loss_coeff and tft_quantile_loss_coeff.")
        if quantile_outputs is None:
            raise RuntimeError("tft_output_mode=joint requires model quantile outputs.")
        point_loss = nn.MSELoss()(outputs, true)
        quantile_loss = self._joint_quantile_criterion(quantile_outputs, true)
        return self._point_loss_coeff * point_loss + self._quantile_loss_coeff * quantile_loss

    def _select_targets_for_loss(self, outputs, batch_y):
        if self._tft_target_positions is None:
            f_dim = -1 if self.args.features == 'MS' else 0
            pred = outputs[:, -self.args.pred_len:, f_dim:]
            true = batch_y[:, -self.args.pred_len:, f_dim:]
        else:
            pred = outputs[:, -self.args.pred_len:, :]
            true = select_tft_truth(batch_y, self.args.pred_len, self._tft_target_positions)
            if pred.shape != true.shape:
                raise RuntimeError(
                    f"TFT prediction/target shape mismatch: pred={tuple(pred.shape)} true={tuple(true.shape)}."
                )
        return pred, true


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_output = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, quantile_outputs, aux_loss = self._extract_outputs_and_aux(model_output)
                pred, true = self._select_targets_for_loss(outputs, batch_y)
                true = true.to(self.device)

                pred = pred.detach()
                true = true.detach()

                detached_quantiles = quantile_outputs.detach() if torch.is_tensor(quantile_outputs) else quantile_outputs
                loss = self._compute_supervised_loss(pred, true, criterion, detached_quantiles)
                loss = combine_primary_and_aux_loss(loss, aux_loss, self._get_aux_loss_coeff())
                if not torch.isfinite(loss):
                    print(f"[vali] Non-finite loss at batch {i}; skipping batch.")
                    continue

                total_loss.append(loss.item())
            total_loss = np.average(total_loss) if total_loss else float('nan')
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, quantile_outputs, aux_loss = self._extract_outputs_and_aux(model_output)
                        outputs, batch_y = self._select_targets_for_loss(outputs, batch_y)
                        batch_y = batch_y.to(self.device)
                        loss = self._compute_supervised_loss(outputs, batch_y, criterion, quantile_outputs)
                        loss = combine_primary_and_aux_loss(loss, aux_loss, self._get_aux_loss_coeff())
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {i + 1}")
                        train_loss.append(loss.item())
                else:
                    model_output = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, quantile_outputs, aux_loss = self._extract_outputs_and_aux(model_output)
                    outputs, batch_y = self._select_targets_for_loss(outputs, batch_y)
                    batch_y = batch_y.to(self.device)
                    loss = self._compute_supervised_loss(outputs, batch_y, criterion, quantile_outputs)
                    loss = combine_primary_and_aux_loss(loss, aux_loss, self._get_aux_loss_coeff())
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {i + 1}")
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        quantile_preds = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_output = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, quantile_outputs, _ = self._extract_outputs_and_aux(model_output)

                outputs, batch_y = self._select_targets_for_loss(outputs, batch_y)
                batch_y = batch_y.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                quantile_outputs_np = quantile_outputs.detach().cpu().numpy() if torch.is_tensor(quantile_outputs) else None
                if test_data.scale and self.args.inverse:
                    if self._tft_target_positions is None:
                        shape = batch_y.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    else:
                        outputs = inverse_transform_selected(outputs, test_data.scaler, self._tft_target_positions)
                        batch_y = inverse_transform_selected(batch_y, test_data.scaler, self._tft_target_positions)
                        if quantile_outputs_np is not None:
                            per_quantile = []
                            for q_idx in range(quantile_outputs_np.shape[2]):
                                restored_q = inverse_transform_selected(
                                    quantile_outputs_np[:, :, q_idx, :],
                                    test_data.scaler,
                                    self._tft_target_positions,
                                )
                                per_quantile.append(restored_q)
                            quantile_outputs_np = np.stack(per_quantile, axis=2)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if quantile_outputs_np is not None:
                    quantile_preds.append(quantile_outputs_np)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    plot_source_pos = -1 if self._tft_target_positions is None else self._tft_target_positions[-1]
                    gt = np.concatenate((input[0, :, plot_source_pos], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, plot_source_pos], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        quantile_summary = None
        if quantile_preds and self._quantile_levels is not None:
            quantile_preds = np.concatenate(quantile_preds, axis=0)
            quantile_preds = quantile_preds.reshape(-1, quantile_preds.shape[-3], quantile_preds.shape[-2], quantile_preds.shape[-1])
            quantile_summary = quantile_metric(quantile_preds, trues, self._quantile_levels)
            f.write(
                ', pinball:{pinball}, coverage:{coverage}, interval_width:{interval_width}, crossing_rate:{crossing_rate}'.format(
                    **quantile_summary
                )
            )
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if quantile_summary is not None:
            with open(folder_path + 'quantile_metrics.json', 'w', encoding='utf-8') as quantile_file:
                json.dump(quantile_summary, quantile_file, indent=2, sort_keys=True)
            np.save(folder_path + 'quantile_pred.npy', quantile_preds)

        return
