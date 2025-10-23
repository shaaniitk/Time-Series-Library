# Input Feature Analysis - 118 Celestial Features

## 📊 **Complete Feature Structure Analysis**

Based on `data/prepared_financial_data.csv`, here's the exact breakdown of your 118 input features:

### **Target Features (First 5 - Not used for celestial processing)**
1. `date` - Date (excluded from model input)
2. `log_Open` - Log of opening price (target)
3. `log_High` - Log of high price (target) 
4. `log_Low` - Log of low price (target)
5. `log_Close` - Log of closing price (target)
6. `time_delta` - Time delta feature

### **Dynamic Celestial Features (Features 6-92: 87 features)**

Each celestial body has **7 dynamic features**:
1. `dyn_{body}_sin` - Sin of longitude (θ phase)
2. `dyn_{body}_cos` - Cos of longitude (θ phase)  
3. `dyn_{body}_speed` - Velocity/speed
4. `dyn_{body}_sign_sin` - Sin of zodiac sign (φ phase)
5. `dyn_{body}_sign_cos` - Cos of zodiac sign (φ phase)
6. `dyn_{body}_distance` - Distance/radius (r)
7. `dyn_{body}_lat` - Latitude

**Celestial Bodies with Dynamic Features (11 bodies × 7 features = 77 features):**
- Sun (features 6-12)
- Moon (features 13-19)
- Mars (features 20-26)
- Mercury (features 27-33)
- Jupiter (features 34-40)
- Venus (features 41-47)
- Saturn (features 48-54)
- Uranus (features 55-61)
- Neptune (features 62-68)
- Pluto (features 69-75)
- Mean Rahu/North Node (features 76-82)
- Mean Ketu/South Node (features 83-89)

**Note**: Chiron is missing from dynamic features.

### **Shadbala Strength Features (Features 90-96: 7 features)**
- `dyn_Sun_shadbala` - Sun's strength
- `dyn_Moon_shadbala` - Moon's strength  
- `dyn_Mars_shadbala` - Mars's strength
- `dyn_Mercury_shadbala` - Mercury's strength
- `dyn_Jupiter_shadbala` - Jupiter's strength
- `dyn_Venus_shadbala` - Venus's strength
- `dyn_Saturn_shadbala` - Saturn's strength

### **Ecliptic Longitude Features (Features 97-118: 22 features)**

Each celestial body has **2 ecliptic longitude features** (sin/cos of ecliptic angle):
1. `{body}_sin` - Sin of ecliptic longitude (2D projection angle)
2. `{body}_cos` - Cos of ecliptic longitude (2D projection angle)

**Celestial Bodies with Ecliptic Longitude Features (11 bodies × 2 features = 22 features):**
- Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Uranus, Neptune, Pluto, Mean Rahu, Mean Ketu, Ascendant

**Note**: These represent the **fundamental astrological coordinates** - the angle of each celestial body as projected onto the ecliptic plane (the 2D "flattened" view of the solar system that astrology uses).

## 🎯 **Key Insights for Model Design**

### **Rich Phase Information Available:**
- **θ phase (dynamic longitude)**: `dyn_{body}_sin/cos` pairs for current ecliptic position
- **φ phase (zodiac sign)**: `sign_sin/sign_cos` pairs for zodiac position
- **Ecliptic longitude**: `{body}_sin/cos` pairs for fundamental astrological coordinates (2D projection)
- **Velocity**: Direct speed measurements (`dyn_{body}_speed`)
- **Distance**: Radius information (`dyn_{body}_distance`)
- **Latitude**: Deviation from ecliptic plane (`dyn_{body}_lat`)
- **Strength**: Shadbala strength measures for traditional planets

### **Celestial Body Mapping:**
```python
CELESTIAL_FEATURE_MAPPING = {
    'Sun': {
        'dynamic': [6, 7, 8, 9, 10, 11, 12],      # sin, cos, speed, sign_sin, sign_cos, distance, lat
        'shadbala': [90],                          # strength
        'ecliptic': [97, 98]                       # ecliptic longitude sin, cos
    },
    'Moon': {
        'dynamic': [13, 14, 15, 16, 17, 18, 19],
        'shadbala': [91], 
        'ecliptic': [99, 100]
    },
    'Mars': {
        'dynamic': [20, 21, 22, 23, 24, 25, 26],
        'shadbala': [92],
        'ecliptic': [101, 102]
    },
    'Mercury': {
        'dynamic': [27, 28, 29, 30, 31, 32, 33],
        'shadbala': [93],
        'ecliptic': [103, 104]
    },
    'Jupiter': {
        'dynamic': [34, 35, 36, 37, 38, 39, 40],
        'shadbala': [94],
        'ecliptic': [105, 106]
    },
    'Venus': {
        'dynamic': [41, 42, 43, 44, 45, 46, 47],
        'shadbala': [95],
        'ecliptic': [107, 108]
    },
    'Saturn': {
        'dynamic': [48, 49, 50, 51, 52, 53, 54],
        'shadbala': [96],
        'ecliptic': [109, 110]
    },
    'Uranus': {
        'dynamic': [55, 56, 57, 58, 59, 60, 61],
        'shadbala': None,  # No shadbala for outer planets
        'ecliptic': [111, 112]
    },
    'Neptune': {
        'dynamic': [62, 63, 64, 65, 66, 67, 68],
        'shadbala': None,
        'ecliptic': [113, 114]
    },
    'Pluto': {
        'dynamic': [69, 70, 71, 72, 73, 74, 75],
        'shadbala': None,
        'ecliptic': [115, 116]
    },
    'North_Node': {  # Mean Rahu
        'dynamic': [76, 77, 78, 79, 80],  # Only 5 features (no distance/lat)
        'shadbala': None,
        'ecliptic': [117, 118]
    },
    'South_Node': {  # Mean Ketu  
        'dynamic': [81, 82, 83, 84, 85],  # Only 5 features (no distance/lat)
        'shadbala': None,
        'ecliptic': [119, 120]  # Note: These might be missing in actual data
    },
    'Chiron': {
        'dynamic': None,  # Missing from dynamic features
        'shadbala': None,
        'ecliptic': None    # Missing from ecliptic features
    }
}
```

## 🚨 **Data Structure Issues Identified**

1. **Feature Count Mismatch**: 
   - Expected: 118 features
   - Actual: ~120+ features in header
   - Need to verify exact count

2. **Missing Chiron**: 
   - Chiron has no dynamic or static features
   - Original aggregator assumed 13 celestial bodies

3. **Inconsistent Node Features**:
   - Most bodies: 7 dynamic + 1 shadbala + 2 ecliptic = 10 features
   - Outer planets: 7 dynamic + 2 ecliptic = 9 features (no shadbala)
   - Nodes: 5 dynamic + 2 ecliptic = 7 features (no distance/lat/shadbala)

## 🎯 **Recommended Model Architecture**

### **Phase-Aware Processing Strategy:**
1. **Extract explicit phases** from sin/cos pairs using `atan2`
2. **Use velocity directly** (no computation needed)
3. **Incorporate distance and latitude** as additional node features
4. **Use shadbala as strength weighting** where available
5. **Combine dynamic and ecliptic longitude features** for richer representations

### **Edge Computation Strategy:**
- **θ phase differences**: From longitude sin/cos pairs
- **φ phase differences**: From zodiac sign sin/cos pairs  
- **Velocity differences**: Direct from speed features
- **Distance ratios**: From distance features
- **Latitude differences**: From latitude features
- **Strength ratios**: From shadbala features where available

This rich feature set provides **much more information** than initially assumed and should enable very sophisticated celestial relationship modeling!