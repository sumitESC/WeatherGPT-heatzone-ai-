<p align="center">
  <img src="https://img.shields.io/badge/🔥_HeatZone_AI-Hackathon_Project-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="HeatZone AI" />
</p>

<h1 align="center">🌡️ HeatZone AI — WeatherGPT</h1>

<p align="center">
  <strong>AI-Powered Conversational Weather Intelligence & Heat Risk Analytics Platform</strong><br/>
  <em>30-Day Multi-Horizon Forecasting • Temporal Fusion Transformer • Zero Hallucination Architecture</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-FastAPI-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/React_18-TypeScript-61DAFB?style=flat-square&logo=react" />
  <img src="https://img.shields.io/badge/ML_Model-TFT_(~15M_params)-FF6F00?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Data-26_Years_ERA5-4285F4?style=flat-square&logo=google-earth" />
  <img src="https://img.shields.io/badge/Coverage-75+_Cities-8BC34A?style=flat-square" />
  <img src="https://img.shields.io/badge/Hackathon-SIH_26068-FFD700?style=flat-square" />
</p>

---

## 🚀 Live Demo

| Platform | Link |
|----------|------|
| 🌐 **GIS Dashboard** | [weathergpt-1d7l.onrender.com](https://weathergpt-1d7l.onrender.com/) |
| 🌐 **Web Application** | [weathergpt-q3w1.onrender.com](https://weathergpt-q3w1.onrender.com/) |
| 📱 **WhatsApp Chatbot** | `+91 88086 41293` *(Send a message to start!)* |
| 📲 **Android App** | Compiled & ready *(Pending Play Store publication)* |

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [ML Models & Methodology](#-ml-models--methodology)
- [Results & Validation](#-results--validation)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [API Endpoints](#-api-endpoints)
- [Screenshots](#-screenshots)
- [Team](#-team)

---

## 🎯 Problem Statement

> **SIH Problem Statement 26068**: Develop a conversational AI platform for weather forecasting to aid disaster management and agricultural planning.

Weather information today is **fragmented** across multiple portals, bulletins, satellite products, and forecast systems — making it nearly impossible for farmers, urban planners, and disaster managers to quickly obtain **actionable insights**. Meanwhile, general-purpose LLMs (like ChatGPT) **hallucinate weather data 39% of the time**, generating plausible but entirely fictitious numbers.

**HeatZone AI solves both problems simultaneously.**

---

## 💡 Our Solution

HeatZone AI introduces a **2-Layer Sense Architecture** that completely decouples data retrieval from language generation:

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│        (Voice / Text / WhatsApp / Dashboard)            │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  LAYER 1: SENSE PIPELINE │
          │  • Intent Classification │
          │  • Entity Extraction     │
          │  • Temporal Resolution   │
          └────────────┬────────────┘
                       │ Structured JSON (deterministic)
          ┌────────────▼─────────────────────┐
          │  DETERMINISTIC DATA RETRIEVAL     │
          │  • TFT 30-Day Forecast Engine     │
          │  • Live Weather APIs (Open-Meteo) │
          │  • Sentinel-2 Satellite Data      │
          │  • Heat Risk Scoring Engine       │
          └────────────┬─────────────────────┘
                       │ Verified numerical data
          ┌────────────▼────────────┐
          │  LAYER 2: ACT PIPELINE   │
          │  • Prompt Injection      │
          │  • Grounded NLG          │
          │  • 0% Hallucination      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────────────┐
          │  RESPONSE DELIVERY               │
          │  (Dashboard / WhatsApp / Voice)   │
          └─────────────────────────────────┘
```

> **Result**: The LLM never invents numbers. It only formats verified data from our ML backend. **0% hallucination rate** across 200 rigorous test queries.

---

## ✨ Key Features

### 🧠 AI & ML Capabilities
- **30-Day (720-hour) Multi-Variable Forecasting** using a custom ~15M parameter Temporal Fusion Transformer
- **35 weather variables predicted simultaneously** (temperature, humidity, wind, UV, precipitation, etc.)
- **Probabilistic forecasting** with P10/P50/P90 quantile uncertainty bands
- **Heat Risk Scoring (0-100)** using Random Forest + Gradient Boosting ensemble (R² = 0.977)
- **Zero hallucination** conversational weather AI via Sense-Layer Architecture

### 🗺️ Dashboard & Visualization
- **Interactive GIS Map** with heat risk overlays across 75+ cities in Uttar Pradesh
- **30-day forecast explorer** with multi-horizon uncertainty visualization
- **City-level deep-dive analytics** with satellite telemetry (NDVI, NDWI, NDBI)
- **Real-time weather monitoring** with live Open-Meteo data integration
- **Embedded AI Chatbot** with voice support (Web Speech API)

### 📡 Edge & Communication
- **WhatsApp Voice/Text Chatbot** for rural accessibility
- **Mass disaster broadcasting** via WhatsApp templates
- **Sub-50ms NLP routing** via Cloudflare Workers
- **Multilingual support**: English, Hindi, Bengali, Marathi

---

## 🏗️ System Architecture

```
                     ┌──────────────────────────────┐
                     │        heatzone-frontend      │
                     │     React 18 + Vite + TS      │
                     │   Dashboard • Map • Chatbot   │
                     └──────────────┬───────────────┘
                                    │ REST API
                     ┌──────────────▼───────────────┐
                     │        heatzone-backend       │
                     │     FastAPI + PyTorch + ML    │
                     └──────────┬────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
     ┌────────────────┐ ┌───────────────┐  ┌──────────────┐
     │  Data Pipeline  │ │  ML Models    │  │  API Routes  │
     │  Open-Meteo     │ │  TFT (~15M)   │  │  /forecast   │
     │  Sentinel-2     │ │  Heatwave RF  │  │  /history    │
     │  ERA5 (26 yrs)  │ │  XGBoost      │  │  /weather    │
     └────────────────┘ └───────────────┘  │  /live-update │
                                           └──────────────┘
```

---

## 🛠️ Tech Stack

### Backend (`heatzone-backend/`)

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI (Python 3.10+) |
| **ML Framework** | PyTorch 2.x |
| **Models** | Temporal Fusion Transformer, Random Forest, XGBoost |
| **Data Sources** | Open-Meteo API, Sentinel-2 (STAC), ERA5 Reanalysis |
| **Deployment** | Render.com (auto-deploy via `render.yaml`) |

### Frontend (`heatzone-frontend/`)

| Category | Technology |
|----------|-----------|
| **Framework** | React 18.3 + TypeScript |
| **Build Tool** | Vite 6.1 |
| **Styling** | Tailwind CSS 4.0 + shadcn/ui (Radix) |
| **Charts** | Recharts 2.15 + Chart.js 4.5 |
| **Maps** | React-Leaflet 4.2 + Leaflet 1.9 |
| **Data Fetching** | TanStack React Query 5.66 |
| **Animations** | Framer Motion 11.18 |

---

## 🤖 ML Models & Methodology

### 1. Temporal Fusion Transformer (TFT) — Primary Forecaster

| Specification | Value |
|--------------|-------|
| **Architecture** | Temporal Fusion Transformer |
| **Parameters** | ~15M+ trainable |
| **Training Data** | 5 GB — 26 years of hourly ERA5 reanalysis (2000–2026) |
| **Total Records** | 15,234,800 hourly rows across 75 cities |
| **Lookback Window** | 168 hours (7 days) |
| **Forecast Horizon** | 720 hours (30 days) |
| **Input Features** | 78 (59 Weather + 13 Satellite/Terrain + 6 Cyclic Temporal) |
| **Output Variables** | 35 weather variables × 720 timesteps |
| **Loss Function** | Quantile Pinball Loss (P10, P50, P90) |
| **Training Hardware** | Google Colab Pro (NVIDIA T4 / A100) |

**Key Components:**
- **Variable Selection Network (VSN)** — Dynamically selects the most relevant features per prediction
- **Static Covariate Encoders** — Processes satellite data (NDVI, NDWI, NDBI, elevation) to understand city geography
- **Interpretable Multi-Head Attention** — Recognizes temporal patterns (diurnal cycles, seasonal trends)
- **Quantile Output Head** — Outputs probabilistic bounds, not single-point estimates

### 2. Heat Risk Analytics — Ensemble Model

| Specification | Value |
|--------------|-------|
| **Algorithm** | VotingRegressor (Gradient Boosting + Random Forest) |
| **Input Features** | 19 (NDVI, NDWI, NDBI, emissions, LST, etc.) |
| **Validation R²** | 0.9771 |
| **Output** | Heat Risk Score (0-100) + Zone Classification (🟢🟡🟠🔴) |

---

## 📊 Results & Validation

### Hallucination Mitigation — 0% vs 39%

| System | Total Queries | Unsupported Numerical Claims (Hallucinations) |
|--------|:------------:|:----------------------------------------------:|
| Direct LLM (Baseline) | 200 | **78 (39.0%)** |
| **HeatZone AI (2-Layer)** | 200 | **0 (0.0%)** ✅ |

### Multi-Horizon Forecast Accuracy (Temperature)

| Horizon | TFT MAE (°C) | Baseline MAE (°C) | Improvement |
|---------|:------------:|:------------------:|:-----------:|
| 24 Hours | 1.12 | 2.30 | **+51.3%** |
| 72 Hours | 1.85 | 3.15 | **+41.2%** |
| 7 Days | 2.40 | 4.50 | **+46.6%** |
| 15 Days | 3.15 | 5.20 | **+39.4%** |
| 30 Days | 3.90 | 5.85 | **+33.3%** |

### Heat Risk Model Performance

| Metric | Value |
|--------|:-----:|
| **R²** | 0.977 |
| **MAE** | 0.81 |
| **RMSE** | 1.05 |

### System Latency

| Component | Median (P50) | P95 |
|-----------|:-----------:|:----:|
| Layer-1 NLP Routing | 14 ms | 22 ms |
| TFT Backend Fetch | 45 ms | 65 ms |
| End-to-End Response | **379 ms** | **497 ms** |

---

## 📁 Repository Structure

```
WeatherGPT-heatzone-ai/
│
├── README.md                          # This file
│
├── heatzone-backend/                  # 🐍 FastAPI ML Server
│   ├── server.py                      # Uvicorn entry point
│   ├── config.py                      # Global configuration
│   ├── requirements.txt               # Python dependencies
│   ├── render.yaml                    # Render.com deployment config
│   ├── update_latest_weather.py       # Live data ingestion CRON script
│   ├── api/
│   │   ├── main.py                    # FastAPI app & routing
│   │   ├── schemas.py                 # Pydantic response models
│   │   └── endpoints/                 # /forecast, /history, /weather, /live-update
│   ├── data_pipeline/
│   │   ├── fetch_openmeteo.py         # Open-Meteo data ingestion
│   │   └── fetch_stac.py             # Sentinel-2 STAC API fetcher
│   └── models/
│       ├── tft_forecaster/            # TFT inference engine
│       │   ├── predict.py             # 30-day multi-variable inference
│       │   └── checkpoints/           # Trained TFT weights (.pt) & scaler.json
│       ├── heatscore/                 # Random Forest Heat Risk model
│       └── predictions/               # Cached CSV outputs
│
└── heatzone-frontend/                 # ⚛️ React Analytics Dashboard
    ├── index.html                     # Entry point
    ├── package.json                   # Node dependencies
    ├── vite.config.ts                 # Vite configuration
    ├── tsconfig.json                  # TypeScript config
    ├── public/                        # Static assets
    └── src/
        ├── App.tsx                    # Root application
        ├── components/
        │   ├── Chatbot.tsx            # Embedded AI chatbot (voice + text)
        │   ├── WeatherGPTHero.tsx     # Landing page hero
        │   ├── SplashScreen.tsx       # Loading experience
        │   └── layout/               # Header, Sidebar, Layout
        ├── pages/
        │   ├── Dashboard.tsx          # City grid with heat risk scores
        │   ├── Forecast.tsx           # 30-day TFT forecast explorer
        │   ├── CityDetail.tsx         # Deep-dive city analytics
        │   ├── MapPage.tsx            # Interactive Leaflet GIS map
        │   ├── History.tsx            # Historical ERA5 trends
        │   ├── Analytics.tsx          # Aggregated analytics
        │   └── Advisor.tsx            # Agricultural advisory
        └── lib/
            ├── renderApi.ts           # API client for backend
            └── llm.ts                 # LLM integration utilities
```

---

## ⚡ Setup & Installation

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

The backend will start at `http://localhost:8000`. Visit `http://localhost:8000/docs` for Swagger API documentation.

### 3. Frontend Setup

```bash
cd heatzone-frontend
npm install
npm run dev
```

The frontend will start at `http://localhost:5173`.

### 4. Environment Variables

**Backend** — No `.env` required for basic local operation.

**Frontend** — Create `.env` in `heatzone-frontend/`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_GROQ_API_KEY=your_groq_api_key
VITE_GROQ_MODEL=llama3-70b-8192
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/forecast/{city}` | 30-day TFT forecast for a city |
| `GET` | `/api/history/{city}` | Historical ERA5 data |
| `GET` | `/api/weather/{city}` | Current live weather |
| `GET` | `/api/live-update` | Trigger data refresh |

### Sample Response — `/api/forecast/Lucknow`

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
    }
  ]
}
```

---

## 🎯 Target Use Cases

| Stakeholder | Use Case |
|-------------|----------|
| 🌾 **Farmers** | 30-day rain probability timelines for irrigation & pesticide scheduling |
| ✈️ **Aviation** | Wind profile briefings, gust forecasts, boundary layer heights |
| 🏙️ **Urban Planners** | Heat risk decomposition (vegetation vs. concrete density) |
| 🚨 **Disaster Management** | Automated mass WhatsApp alerts for extreme weather anomalies |

---

## 📚 References

1. Lim, B., et al. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. International Journal of Forecasting.
2. Hersbach, H., et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society.
3. Open-Meteo API: [https://open-meteo.com/](https://open-meteo.com/)
4. Copernicus Data Space Ecosystem (Sentinel-2): [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
5. Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS.

---

## 👨‍💻 Team

**Sumit Kushwaha**
- 📧 Email: [iamkussumit@gmail.com](mailto:iamkussumit@gmail.com)
- 📞 Contact: +91 9616550356

---

<p align="center">
  <strong>Built with ❤️ for Smart India Hackathon 2026</strong><br/>
  <em>Problem Statement 26068 — Conversational AI for Weather Intelligence</em>
</p>
