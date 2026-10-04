<p align="center">
  <img src="https://img.shields.io/badge/HeatZone_AI-Hackathon_Project-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="HeatZone AI" />
</p>

<h1 align="center">HeatZone AI</h1>

<p align="center">
  <strong>AI-Powered Urban Heat Risk Analytics & 30-Day Multi-Horizon Weather Forecasting Platform</strong><br/>
  <em>Temporal Fusion Transformer | Satellite Intelligence | Real-Time Heat Risk Scoring | GIS Dashboard</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-FastAPI-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/React_18-TypeScript-61DAFB?style=flat-square&logo=react" />
  <img src="https://img.shields.io/badge/ML-TFT_(~15M_params)-FF6F00?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Data-26_Years_ERA5_(5GB)-4285F4?style=flat-square&logo=google-earth" />
  <img src="https://img.shields.io/badge/Satellite-Sentinel--2_L2A-1B5E20?style=flat-square" />
  <img src="https://img.shields.io/badge/Coverage-75+_Cities_UP-8BC34A?style=flat-square" />
  <img src="https://img.shields.io/badge/Hackathon-SIH_2026-FFD700?style=flat-square" />
</p>

---

## Live Demo

| Platform | Link |
|----------|------|
| **HeatZone Dashboard** | [Live on Render](https://weathergpt-q3w1.onrender.com/) |
| **Backend API (Swagger)** | [API Docs](https://heatzone-backend.onrender.com/docs) |

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [System Architecture](#system-architecture)
- [HeatZone Backend — ML Server](#heatzone-backend--ml-server)
- [HeatZone Frontend — Analytics Dashboard](#heatzone-frontend--analytics-dashboard)
- [ML Models & Methodology](#ml-models--methodology)
- [Dataset & Feature Engineering](#dataset--feature-engineering)
- [Results & Validation](#results--validation)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [API Endpoints](#api-endpoints)
- [Team](#team)

---

## Problem Statement

India's rapid urbanization is creating **deadly Urban Heat Islands (UHIs)** across cities in Uttar Pradesh. Concrete-heavy infrastructure, shrinking green cover, and rising emissions are driving surface temperatures 5-10 C above surrounding rural areas. **Heatwaves killed over 2,500 people in India in 2023 alone.**

Current challenges:
- No city-level heat risk scoring system exists for UP's 75 districts
- Weather forecasts are limited to 7 days — inadequate for agricultural & urban planning
- Satellite data (NDVI, NDBI, LST) is siloed and not integrated with weather models
- No platform combines ML forecasting with real-time heat vulnerability analytics

**HeatZone AI was built to fix this.**

---

## Our Solution

HeatZone AI is a **full-stack AI platform** consisting of two core modules:

### `heatzone-backend` — The ML Intelligence Engine
A FastAPI server that runs a custom **~15M parameter Temporal Fusion Transformer (TFT)** trained on **26 years of ERA5 reanalysis data + Sentinel-2 satellite imagery**. It generates **30-day (720-hour) multi-variable weather forecasts** and computes proprietary **Heat Risk Scores (0-100)** for every city.

### `heatzone-frontend` — The Analytics Dashboard
A production React 18 application providing **interactive GIS maps, multi-horizon forecast explorers, satellite telemetry visualizations**, and city-level heat vulnerability analytics — enabling disaster managers, urban planners, and agricultural officers to make data-driven decisions.

```
 +-------------------------------------------------------------+
 |                   heatzone-frontend                          |
 |          React 18 + Vite + TypeScript + Tailwind             |
 |                                                              |
 |  +----------+ +-----------+ +----------+ +--------------+   |
 |  |Dashboard | | Forecast  | | GIS Map  | | City Detail  |   |
 |  |75 Cities | | 30-Day    | | Leaflet  | | Satellite    |   |
 |  |Risk Grid | | Explorer  | | Heatmap  | | Analytics    |   |
 |  +----+-----+ +-----+-----+ +----+-----+ +------+-------+   |
 |       +-------------------+------+---------------+           |
 +---------------------------+------+--------------------------+
                             | REST API
 +---------------------------v------v--------------------------+
 |                    heatzone-backend                          |
 |              FastAPI + PyTorch + ML Models                   |
 |                                                              |
 |  +--------------+  +--------------+  +-------------------+   |
 |  | Data Pipeline |  |  TFT Model   |  | Heat Risk Engine  |  |
 |  | Open-Meteo    |  |  ~15M params |  | RF + GradBoost    |  |
 |  | Sentinel-2    |  |  720hr fcast |  | Score 0-100       |  |
 |  | ERA5 (26 yrs) |  |  35 variables|  | R2 = 0.977        |  |
 |  +--------------+  +--------------+  +-------------------+   |
 +-------------------------------------------------------------+
```

---

## System Architecture

```
                    +----------------------------+
                    |    FastAPI REST Server      |
                    |      (server.py)            |
                    +-----------+----------------+
                                |
            +-------------------+-------------------+
            v                   v                   v
   +--------------+    +----------------+    +-------------+
   |  Data        |    |   ML Models    |    |    API      |
   |  Pipeline    |    |  Inference     |    |  Routers    |
   +------+-------+    +-------+--------+    +------+------+
          |                     |                    |
  +-------+-------+    +-------+----------+   +-----+------------------+
  | Open-Meteo    |    | TFT Forecaster   |   | /forecast/{city}       |
  | Sentinel-2    |    | Heatwave RF      |   | /history/{city}        |
  | ERA5 (past)   |    | XGBoost Fallback |   | /weather/{city}        |
  +---------------+    +------------------+   | /live-update           |
                                              +------------------------+
```

---

## HeatZone Backend — ML Server

The backend is the **core computational engine** of HeatZone AI. It is responsible for serving multi-horizon numerical forecasts, executing data ingestion pipelines, computing heat risk scores, and serving meteorological intelligence via RESTful APIs.

### Core Capabilities

| Capability | Description |
|-----------|-------------|
| **30-Day Forecasting** | Generates 720-hour multi-variable predictions using the TFT model |
| **Heat Risk Scoring** | Computes city-level Heat Risk Score (0-100) with zone classification (LOW / MODERATE / HIGH / EXTREME) |
| **Live Data Ingestion** | CRON-based pipelines fetching real-time data from Open-Meteo for 75+ cities |
| **Satellite Processing** | Ingests Sentinel-2 L2A indices (NDVI, NDWI, NDBI, SAVI, BSI, Albedo) via STAC API |
| **Model Inference** | Serves PyTorch TFT model weights with configurable batch inference |
| **XGBoost Fallback** | Secondary model for rapid predictions when TFT latency is too high |

### Backend Directory Structure

```
heatzone-backend/
├── server.py                    # Uvicorn entry point
├── config.py                    # Global configuration (cities, coordinates, API keys)
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render.com deployment configuration
├── update_latest_weather.py     # Live data ingestion CRON script
│
├── api/
│   ├── main.py                  # FastAPI app factory & CORS middleware
│   ├── schemas.py               # Pydantic response models
│   └── endpoints/               # Route handlers
│       ├── forecast.py          # /forecast/{city} — TFT 30-day predictions
│       ├── history.py           # /history/{city} — ERA5 historical trends
│       ├── weather.py           # /weather/{city} — Current conditions
│       └── sat_model.py         # /sat-model — Satellite model inference
│
├── data_pipeline/
│   ├── fetch_openmeteo.py       # Open-Meteo API ingestion (59 variables)
│   └── fetch_stac.py            # Sentinel-2 STAC API satellite data fetcher
│
├── models/
│   ├── tft_forecaster/          # Temporal Fusion Transformer inference
│   │   ├── predict.py           # 30-day multi-variable inference wrapper
│   │   └── checkpoints/         # Trained TFT weights (.pt) & scaler.json
│   ├── heatscore/               # Heat Risk scoring engine
│   │   ├── heatwave_model.py    # Random Forest + Gradient Boosting ensemble
│   │   └── model_metadata.json  # Feature importance & hyperparameters
│   ├── sat_model/               # Satellite-based prediction model
│   │   ├── model.py             # Neural network architecture
│   │   ├── dataset.py           # Data loader for satellite features
│   │   ├── train.py             # Training loop
│   │   └── checkpoints/         # Trained weights (best_model.pt)
│   ├── xgboost/                 # XGBoost fallback forecaster
│   │   └── model/
│   │       ├── xgb_model.py     # XGBoost inference
│   │       ├── data_pipeline.py # Feature engineering
│   │       └── context_builder.py # Context window construction
│   └── predictions/             # Cached CSV prediction outputs per city
│
└── data/                        # Local data storage
    ├── historical/              # ERA5 + Open-Meteo historical archives
    ├── satellite/               # Sentinel-2 processed indices
    └── heatscores/              # Pre-computed heat risk scores
```

### Heat Risk Scoring Algorithm

The Heat Risk Score (0-100) is a **composite index** computed from multiple environmental and urban factors:

```
Heat Risk Score = f(
    Temperature_2m,           # Current & forecasted air temperature
    Land_Surface_Temp,        # MODIS-derived LST
    NDBI,                     # Built-up density (concrete/asphalt)
    NDVI,                     # Vegetation cover (cooling factor)
    Emission_Index,           # Industrial & vehicular emissions
    Population_Density,       # Human vulnerability factor
    Urban_Nightlight_Glow,    # VIIRS nighttime urban intensity
    Albedo                    # Surface reflectivity
)
```

| Zone | Score Range | Meaning |
|------|:-----------:|---------|
| **LOW** | 0 - 25 | Safe conditions, adequate green cover |
| **MODERATE** | 25 - 50 | Elevated risk, monitor vulnerable populations |
| **HIGH** | 50 - 75 | Dangerous, advisory for outdoor workers |
| **EXTREME** | 75 - 100 | Severe heat emergency, immediate intervention needed |

---

## HeatZone Frontend — Analytics Dashboard

The frontend is a **production-grade React 18 application** that serves as the visual analytics layer, consuming data from the HeatZone ML API to provide interactive insights for stakeholders.

### Dashboard Pages

#### 1. Dashboard (`Dashboard.tsx`)
The main overview showing **75 city cards** with:
- Live heat risk scores (0-100) with color-coded zone badges
- Current temperature readings from Open-Meteo
- Active severe weather alerts based on TFT forecast thresholds
- Search and filter capabilities across all UP districts

#### 2. 30-Day Forecast Explorer (`Forecast.tsx`)
Dedicated visualization for TFT model outputs:
- **Multi-horizon area charts** displaying 720 hours of predictions for Temperature, Precipitation, Humidity, Wind Speed, UV Index
- **Quantile uncertainty bands** showing P10/P50/P90 probabilistic bounds — uncertainty widens as the horizon extends, which is scientifically accurate
- **Variable comparison** across multiple weather parameters

#### 3. City Detail (`CityDetail.tsx`)
Deep-dive analytics for a single city including:
- **Satellite Telemetry Panel**: Displays Sentinel-2 indices — NDVI (vegetation), NDWI (water), NDBI (built-up) — used as static features by the TFT model
- **Heat Risk Decomposition**: Breaks down the score into primary drivers (e.g., "Concrete Density contributed 23 points", "Vegetation Scarcity penalized by 15 points")
- **7-day hourly forecast timeline** with detailed metrics

#### 4. Interactive GIS Map (`MapPage.tsx`)
React-Leaflet geospatial visualization:
- **Heat risk heatmap overlay** across Uttar Pradesh with color-coded city markers
- **Click-to-navigate**: Pins link directly to City Detail views
- **Layer toggles**: Switch between temperature, heat risk, precipitation, and satellite overlays

#### 5. Historical Trends (`History.tsx`)
ERA5 climate analysis:
- **26-year historical climate trends** per city (2000-2026)
- **Year-over-year temperature anomaly tracking**
- **Seasonal pattern visualization** for agricultural planning

#### 6. Analytics (`Analytics.tsx`)
Aggregated analytics across all 75 cities:
- **Statewide heat risk distribution** charts
- **Top-10 most vulnerable cities** ranked by heat risk score
- **Satellite index comparisons** across districts

#### 7. Agricultural Advisor (`Advisor.tsx`)
Decision support for agricultural stakeholders:
- **Crop-specific weather advisories** based on 30-day forecasts
- **Irrigation scheduling recommendations** using precipitation probability timelines
- **Fertilizer application windows** based on wind speed and rain forecasts

### Frontend Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Framework** | React 18.3 + TypeScript | Type-safe component architecture |
| **Build Tool** | Vite 6.1 | Fast HMR & optimized production builds |
| **Styling** | Tailwind CSS 4.0 + shadcn/ui (Radix) | Consistent design system |
| **Charts** | Recharts 2.15 + Chart.js 4.5 | Multi-horizon forecast visualization |
| **Maps** | React-Leaflet 4.2 + Leaflet 1.9 | GIS heat risk mapping |
| **Data Fetching** | TanStack React Query 5.66 | Cached API calls with auto-refresh |
| **Animations** | Framer Motion 11.18 | Smooth page transitions & micro-interactions |

---

## ML Models & Methodology

### 1. Temporal Fusion Transformer (TFT) — Primary Forecaster

The TFT is the backbone of HeatZone's forecasting engine. It was specifically chosen over standard LSTMs and ARIMA because it can:
- **Selectively weigh different features at different timesteps** via Variable Selection Networks
- **Cleanly separate static metadata** (elevation, NDBI) **from dynamic data** (temperature, wind)
- **Output probabilistic bounds** instead of brittle single-point estimates

| Specification | Value |
|--------------|-------|
| **Architecture** | Temporal Fusion Transformer |
| **Trainable Parameters** | ~15M+ |
| **Training Data** | 5 GB — 26 years hourly ERA5 + Sentinel-2 (2000-2026) |
| **Total Records** | 15,234,800 hourly rows across 75 cities |
| **Lookback Window** | 168 hours (7 days of historical context) |
| **Forecast Horizon** | 720 hours (30 days into the future) |
| **Input Features** | 78 total (59 Weather + 13 Satellite/Terrain + 6 Cyclic Temporal) |
| **Output** | 35 weather variables x 720 timesteps x 3 quantiles |
| **d_model** | 128 (local) / 192 (Colab) |
| **Attention Heads** | 4 (local) / 6 (Colab) |
| **LSTM Layers** | 2 |
| **Loss Function** | Quantile Pinball Loss (P10, P50, P90) |
| **Optimizer** | AdamW (lr=1.5e-3, weight_decay=1e-5) |
| **Training Hardware** | Google Colab Pro (NVIDIA T4 / A100 GPUs) |

#### TFT Architectural Components

1. **Static Covariate Encoders**: 13 satellite/terrain features (NDVI, NDWI, NDBI, SAVI, BSI, Albedo, Elevation) pass through Gated Residual Networks (GRNs) to create a static context vector. If a city has high NDBI (lots of concrete), the network conditions itself to predict higher temperatures.

2. **Variable Selection Network (VSN)**: Uses softmax attention to dynamically weight the 59 weather features. When predicting rainfall, it learns to heavily weigh CAPE, humidity, and cloud cover while ignoring deep soil temperature.

3. **Sequence-to-Sequence LSTM Encoders**: Processes the 168-hour historical sequence. Static context vectors initialize the cell states, making temporal processing geography-aware.

4. **Interpretable Multi-Head Attention**: Transformer attention over LSTM outputs recognizes patterns like diurnal temperature cycles (24h periodicity) and synoptic weather events.

5. **Quantile Output Head**: Outputs P10/P50/P90 bounds per variable. The uncertainty naturally widens at Day 30 vs Day 1, accurately reflecting the limits of atmospheric predictability.

#### Prediction Targets (35 Variables)

- **Thermal**: Temperature, Dewpoint, Apparent Temp, Wet Bulb Globe Temperature
- **Hydrological**: Humidity, Precipitation, Rain Probability, ET0, Vapour Pressure Deficit
- **Pressure & Wind**: MSL Pressure, Wind Speed (10m, 80m, 100m, 120m, 180m), Wind Direction, Gusts
- **Solar & Storm**: Shortwave Radiation, UV Index, CAPE (Convective Available Potential Energy)
- **Soil**: Soil Temperature & Moisture at 4 depths (0-7cm, 7-28cm, 28-100cm, 100-255cm)

### 2. Heat Risk Analytics — Ensemble Model

| Specification | Value |
|--------------|-------|
| **Algorithm** | VotingRegressor (Gradient Boosting + Random Forest) |
| **Input Features** | 19 (NDVI, NDWI, NDBI, emissions, LST, population density, etc.) |
| **Validation R2** | **0.9771** |
| **MAE** | 0.81 points (on 0-100 scale) |
| **RMSE** | 1.05 points |
| **Output** | Heat Risk Score (0-100) + Zone (LOW / MODERATE / HIGH / EXTREME) + Causal Explanation |

### 3. Satellite-Based Prediction Model

A dedicated neural network model that processes Sentinel-2 spectral data to predict environmental indicators:
- Trained on L2A surface reflectance bands (Blue, Green, Red, NIR, SWIR)
- Generates derived indices used as static features for the TFT

### 4. XGBoost Fallback

A lightweight gradient boosting model that provides rapid predictions when TFT inference latency is too high or GPU resources are constrained. Trades accuracy for speed.

---

## Dataset & Feature Engineering

### Data Sources

| Source | Type | Temporal Coverage | Variables |
|--------|------|-------------------|-----------|
| **ERA5 Reanalysis** (Copernicus CDS) | Historical atmospheric | 2000-2026 (26 years) | 6 core reanalysis variables |
| **Open-Meteo Historical** | Hourly weather archives | 2000-2026 | 59 granular weather variables |
| **Sentinel-2 L2A** (Microsoft Planetary Computer) | Satellite imagery | Per-city snapshots | 5 spectral bands -> 7 derived indices |
| **Copernicus DEM GLO-30** | Digital Elevation Model | Static | Elevation per city grid |

### Dataset Scale

| Metric | Value |
|--------|-------|
| **Temporal Span** | January 1, 2000 to September 20, 2026 |
| **Resolution** | Hourly |
| **Spatial Coverage** | 75 cities across Uttar Pradesh |
| **Total Records** | **15,234,800** hourly rows |
| **Storage** | ~5 GB processed Parquet, partitioned by city |

### Feature Composition (78 Total Features)

**Weather Variables (59):**
- Thermal: `temperature_2m`, `dew_point_2m`, `apparent_temperature`, `wet_bulb_temperature_2m`, `vapour_pressure_deficit`
- Hydrological: `relative_humidity_2m`, `precipitation`, `rain`, `snowfall`, `et0_fao_evapotranspiration`
- Pressure/Wind: `pressure_msl`, `surface_pressure`, wind speed/direction at 5 heights, `wind_gusts_10m`
- Soil: Temperature & moisture at 4 depths
- Radiation: `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`, `direct_normal_irradiance`
- Atmospheric: `boundary_layer_height`, `cape`, `lifted_index`, `freezing_level_height`

**Satellite & Terrain (13):**
- NDVI (vegetation health), NDWI (water content), NDBI (built-up/concrete density)
- SAVI (soil-adjusted vegetation), BSI (bare soil exposure), Albedo (reflectivity)
- Elevation (meters above sea level)

**Cyclic Temporal (6):**
- `hour_sin/cos` (period=24), `doy_sin/cos` (period=365.25), `month_sin/cos` (period=12)

---

## Results & Validation

### Multi-Horizon Forecast Accuracy (Temperature)

| Horizon | TFT MAE (C) | TFT RMSE | Baseline MAE (C) | Improvement |
|---------|:------------:|:--------:|:------------------:|:-----------:|
| **24 Hours (Day 1)** | 1.12 | 1.45 | 2.30 | **+51.3%** |
| **72 Hours (Day 3)** | 1.85 | 2.21 | 3.15 | **+41.2%** |
| **7 Days** | 2.40 | 3.10 | 4.50 | **+46.6%** |
| **15 Days** | 3.15 | 3.95 | 5.20 | **+39.4%** |
| **30 Days (720h)** | 3.90 | 4.80 | 5.85 | **+33.3%** |

> The TFT significantly outperforms the persistence baseline across all horizons. The probabilistic P10-P90 bands correctly widen at longer horizons, accurately modeling the limits of atmospheric predictability.

### Heat Risk Model Performance

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| **R2** | **0.977** | Captures 97.7% of variance in heat risk |
| **MAE** | 0.81 | Less than 1 point error on 0-100 scale |
| **RMSE** | 1.05 | Extremely tight predictions |

### Backend API Latency

| Component | Median (P50) | P95 | P99 |
|-----------|:-----------:|:----:|:----:|
| TFT Forecast Fetch | 45 ms | 65 ms | 110 ms |
| Heat Risk Computation | 12 ms | 18 ms | 30 ms |
| Full API Response | **57 ms** | **83 ms** | **140 ms** |

### Sample Prediction Output

```json
{
  "city": "Lucknow",
  "start_date": "2026-09-24 00:00:00",
  "forecast": [
    {
      "timestamp": "2026-09-24 14:00:00",
      "pred_temperature_2m": 34.2,
      "pred_relative_humidity": 58.9,
      "pred_precipitation": 0.0,
      "pred_wind_speed_10m": 6.45,
      "pred_uv_index": 9.8,
      "heat_risk_score": 72.4,
      "heat_zone": "HIGH"
    },
    {
      "timestamp": "2026-09-25 18:00:00",
      "pred_temperature_2m": 29.7,
      "pred_relative_humidity": 72.1,
      "pred_precipitation": 2.4,
      "pred_wind_speed_10m": 4.88,
      "pred_uv_index": 1.2,
      "heat_risk_score": 41.2,
      "heat_zone": "MODERATE"
    }
  ]
}
```

---

## Target Use Cases

| Stakeholder | How HeatZone Helps |
|-------------|--------------------|
| **Urban Planners** | Identify heat-vulnerable zones using NDBI/NDVI decomposition. Quantify the cooling impact of adding green corridors |
| **Disaster Managers** | Monitor 75 cities in real-time. Get 30-day advance warning of heatwave events via TFT forecasts |
| **Agricultural Officers** | 30-day precipitation timelines for irrigation planning. Fertilizer application windows based on wind/rain predictions |
| **Public Health** | Heat risk zone alerts for hospitals and emergency services. Track vulnerable districts approaching EXTREME (75+) scores |
| **Infrastructure** | Soil temperature forecasts at 4 depths for construction planning. Wind speed predictions at multiple heights for structural assessments |

---

## Repository Structure

```
WeatherGPT-heatzone-ai/
|
├── README.md                          # This file
|
├── heatzone-backend/                  # FastAPI ML Server
│   ├── server.py                      # Uvicorn entry point
│   ├── config.py                      # Global configuration
│   ├── requirements.txt               # Python dependencies
│   ├── render.yaml                    # Render.com deployment config
│   ├── update_latest_weather.py       # Live data ingestion CRON script
│   ├── api/                           # REST API layer
│   │   ├── main.py                    # FastAPI app & routing
│   │   ├── schemas.py                 # Pydantic response models
│   │   └── endpoints/                 # /forecast, /history, /weather, /live-update
│   ├── data_pipeline/                 # Data ingestion services
│   │   ├── fetch_openmeteo.py         # Open-Meteo data ingestion (59 vars)
│   │   └── fetch_stac.py             # Sentinel-2 STAC API satellite fetcher
│   ├── data/                          # Local data storage (historical, satellite, heatscores)
│   └── models/                        # ML model zoo
│       ├── tft_forecaster/            # TFT inference engine + checkpoints
│       ├── heatscore/                 # Random Forest Heat Risk model
│       ├── sat_model/                 # Satellite-based neural network
│       ├── xgboost/                   # XGBoost fallback forecaster
│       └── predictions/               # Cached CSV prediction outputs
|
└── heatzone-frontend/                 # React Analytics Dashboard
    ├── index.html                     # Entry point
    ├── package.json                   # Node dependencies
    ├── vite.config.ts                 # Vite configuration
    ├── tsconfig.json                  # TypeScript config
    ├── public/                        # Static assets & images
    ├── mock-backend/                  # Express mock server for offline dev
    └── src/
        ├── App.tsx                    # Root application with routing
        ├── components/
        │   ├── HeatZoneBadge.tsx       # Color-coded risk zone badge
        │   ├── StatCard.tsx            # Metric display card
        │   ├── SplashScreen.tsx        # Loading experience
        │   ├── BackendWakingOverlay.tsx # Backend cold-start handler
        │   └── layout/                # Header, Sidebar, Layout shells
        ├── pages/
        │   ├── Dashboard.tsx          # 75-city grid with heat risk scores
        │   ├── Forecast.tsx           # 30-day TFT forecast explorer
        │   ├── CityDetail.tsx         # City deep-dive + satellite telemetry
        │   ├── MapPage.tsx            # Interactive Leaflet GIS heat map
        │   ├── History.tsx            # 26-year ERA5 climate trends
        │   ├── Analytics.tsx          # Statewide aggregated analytics
        │   └── Advisor.tsx            # Agricultural decision support
        ├── lib/
        │   ├── renderApi.ts           # API client for HeatZone backend
        │   └── utils.ts              # Utility functions
        └── context/                   # React context providers
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/sumitESC/WeatherGPT-heatzone-ai-.git
cd WeatherGPT-heatzone-ai-
```

### 2. Backend Setup

```bash
cd heatzone-backend
python -m venv venv

# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
python server.py
```

- **Server**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd heatzone-frontend
npm install
npm run dev
```

- **Dashboard**: `http://localhost:5173`

### 4. Environment Variables

**Frontend** — Create `.env` in `heatzone-frontend/`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_GROQ_API_KEY=your_groq_api_key
VITE_GROQ_MODEL=llama3-70b-8192
```

### 5. Production Deployment

Both services are configured for **Render.com**:
- Backend auto-deploys via `render.yaml` with background CRON tasks for continuous data ingestion
- Frontend builds to static files deployable on Render, Vercel, or Cloudflare Pages

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/forecast/{city}` | 30-day (720hr) TFT forecast with heat risk |
| `GET` | `/api/history/{city}` | Historical ERA5 climate data |
| `GET` | `/api/weather/{city}` | Current live weather from Open-Meteo |
| `GET` | `/api/live-update` | Trigger fresh data ingestion |
| `GET` | `/api/sat-model/{city}` | Satellite model inference results |

---

## References

1. Lim, B., et al. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. International Journal of Forecasting, 37(4), 1748-1764.
2. Hersbach, H., et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
3. Open-Meteo API: [https://open-meteo.com/](https://open-meteo.com/)
4. Copernicus Data Space Ecosystem (Sentinel-2): [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
5. Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS.

---

## Team

**Sumit Kushwaha**
- Email: [iamkussumit@gmail.com](mailto:iamkussumit@gmail.com)
- Contact: +91 9616550356

---

<p align="center">
  <strong>Built for Smart India Hackathon 2026</strong><br/>
  <em>HeatZone AI — Protecting communities from extreme heat through data-driven intelligence</em>
</p>
