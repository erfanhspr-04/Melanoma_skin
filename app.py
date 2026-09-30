import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
import time
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="سامانه تشخیص ملانوما",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "saved_models/EfficientNetB3_final.keras"
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.5

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS + 3D ANIMATED BACKGROUND
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

/* ══ reset & base ══ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], * { font-family: 'Vazirmatn', sans-serif !important; }

/* ══ CSS variables ══ */
:root {
  --teal:    #00d4aa;
  --teal2:   #00ffcc;
  --teal3:   #007a62;
  --blue:    #38bdf8;
  --navy:    #020b18;
  --navy2:   #041428;
  --navy3:   #061e38;
  --txt:     #e2f0ff;
  --muted:   #5a8aaa;
  --danger:  #f87171;
  --success: #00d4aa;
  --warn:    #fbbf24;
  --glass:   rgba(4, 20, 40, 0.72);
  --border:  rgba(0, 212, 170, 0.15);
  --border2: rgba(0, 212, 170, 0.30);
  --glow:    rgba(0, 212, 170, 0.18);
}

/* ══ full-page animated background ══ */
.stApp {
  background: var(--navy) !important;
  color: var(--txt);
  overflow-x: hidden;
}

/* ══ floating orbs ══ */
.orb-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  opacity: 0.22;
}
.orb-1 {
  width: 600px; height: 600px;
  background: radial-gradient(circle, #00d4aa 0%, transparent 65%);
  top: -150px; left: -150px;
  animation: orb1 20s ease-in-out infinite alternate;
}
.orb-2 {
  width: 500px; height: 500px;
  background: radial-gradient(circle, #38bdf8 0%, transparent 65%);
  top: 20%; right: -120px;
  animation: orb2 26s ease-in-out infinite alternate;
  opacity: 0.16;
}
.orb-3 {
  width: 450px; height: 450px;
  background: radial-gradient(circle, #a78bfa 0%, transparent 65%);
  bottom: -100px; left: 25%;
  animation: orb3 32s ease-in-out infinite alternate;
  opacity: 0.14;
}
.orb-4 {
  width: 320px; height: 320px;
  background: radial-gradient(circle, #00d4aa 0%, transparent 65%);
  bottom: 15%; right: 5%;
  animation: orb4 18s ease-in-out infinite alternate;
  opacity: 0.13;
}
.orb-5 {
  width: 250px; height: 250px;
  background: radial-gradient(circle, #f87171 0%, transparent 65%);
  top: 55%; left: 5%;
  animation: orb5 24s ease-in-out infinite alternate;
  opacity: 0.08;
}
@keyframes orb1 {
  0%   { transform: translate(0px, 0px) scale(1); }
  100% { transform: translate(80px, 60px) scale(1.15); }
}
@keyframes orb2 {
  0%   { transform: translate(0px, 0px) scale(1); }
  100% { transform: translate(-60px, 80px) scale(1.1); }
}
@keyframes orb3 {
  0%   { transform: translate(0px, 0px) scale(1); }
  100% { transform: translate(50px, -70px) scale(1.12); }
}
@keyframes orb4 {
  0%   { transform: translate(0px, 0px) scale(1); }
  100% { transform: translate(-40px, -50px) scale(1.08); }
}
@keyframes orb5 {
  0%   { transform: translate(0px, 0px) scale(1); }
  100% { transform: translate(60px, -40px) scale(1.1); }
}

/* ══ CSS particle dots ══ */
.particles-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.p-dot {
  position: absolute;
  border-radius: 50%;
  animation: p-float linear infinite;
}
@keyframes p-float {
  0%   { transform: translateY(100vh) scale(0); opacity: 0; }
  5%   { opacity: 1; }
  95%  { opacity: 0.8; }
  100% { transform: translateY(-10vh) scale(1); opacity: 0; }
}

/* ══ CSS spinning rings (corner decorations) ══ */
.rings-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.ring {
  position: absolute;
  border-radius: 50%;
  border: 1px solid rgba(0,212,170,0.12);
  animation: ring-spin linear infinite;
}
.ring-tl-1 { width:260px; height:130px; top:-40px; left:-60px; animation-duration:12s; border-color:rgba(0,212,170,0.10); }
.ring-tl-2 { width:180px; height:90px;  top:-20px; left:-30px; animation-duration:8s; animation-direction:reverse; border-color:rgba(0,212,170,0.14); }
.ring-tl-3 { width:100px; height:50px;  top:5px;   left:5px;   animation-duration:5s; border-color:rgba(56,189,248,0.16); }
.ring-br-1 { width:280px; height:140px; bottom:-50px; right:-70px; animation-duration:14s; border-color:rgba(0,212,170,0.09); }
.ring-br-2 { width:190px; height:95px;  bottom:-25px; right:-35px; animation-duration:9s; animation-direction:reverse; border-color:rgba(0,212,170,0.13); }
.ring-br-3 { width:110px; height:55px;  bottom:5px;   right:5px;   animation-duration:6s; border-color:rgba(167,139,250,0.15); }
.ring-tr-1 { width:220px; height:110px; top:-30px; right:-50px; animation-duration:16s; animation-direction:reverse; border-color:rgba(56,189,248,0.08); }
.ring-tr-2 { width:140px; height:70px;  top:-10px; right:-20px; animation-duration:10s; border-color:rgba(56,189,248,0.12); }
.ring-bl-1 { width:200px; height:100px; bottom:-30px; left:-40px; animation-duration:13s; border-color:rgba(167,139,250,0.09); }
.ring-bl-2 { width:130px; height:65px;  bottom:-10px; left:-15px; animation-duration:8s; animation-direction:reverse; border-color:rgba(167,139,250,0.13); }
@keyframes ring-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

/* ══ grid overlay ══ */
.grid-overlay {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-image:
    linear-gradient(rgba(0,212,170,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,170,0.035) 1px, transparent 1px);
  background-size: 55px 55px;
  -webkit-mask-image: radial-gradient(ellipse 85% 85% at 50% 50%, black 30%, transparent 100%);
  mask-image: radial-gradient(ellipse 85% 85% at 50% 50%, black 30%, transparent 100%);
}

/* ══ moving gradient sweep ══ */
.sweep-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background: conic-gradient(
    from 0deg at 50% 50%,
    transparent 0deg,
    rgba(0,212,170,0.03) 60deg,
    transparent 120deg
  );
  animation: sweep-rotate 20s linear infinite;
}
@keyframes sweep-rotate {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

/* ══ pulsing center glow ══ */
.center-glow {
  position: fixed;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 800px; height: 400px;
  background: radial-gradient(ellipse, rgba(0,212,170,0.04) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
  animation: center-pulse 6s ease-in-out infinite;
}
@keyframes center-pulse {
  0%,100% { opacity: 0.5; transform: translate(-50%,-50%) scale(1); }
  50%      { opacity: 1;   transform: translate(-50%,-50%) scale(1.15); }
}

/* ══ raise content above bg ══ */
.stApp > div { position: relative; z-index: 1; }
.block-container {
  position: relative; z-index: 1 !important;
  max-width: 820px !important;
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}

/* ══ TOPBAR ══ */
.topbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 0 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-icon {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--teal), #007a62);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  box-shadow: 0 0 20px rgba(0,212,170,0.4);
  animation: icon-pulse 3s ease-in-out infinite;
}
@keyframes icon-pulse {
  0%,100% { box-shadow: 0 0 16px rgba(0,212,170,0.35); }
  50%      { box-shadow: 0 0 32px rgba(0,212,170,0.7); }
}
.brand-name { font-size: 15px; font-weight: 700; color: var(--teal); letter-spacing: .06em; }
.brand-ver  { font-size: 11px; color: var(--muted); margin-top: 1px; }
.status-pill {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(0,212,170,0.07);
  border: 1px solid rgba(0,212,170,0.22);
  border-radius: 999px; padding: 5px 14px;
  font-size: 12px; font-weight: 600; color: var(--teal);
}
.s-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--teal);
  animation: s-pulse 2s infinite;
}
@keyframes s-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,212,170,.7); }
  50%      { box-shadow: 0 0 0 5px rgba(0,212,170,0); }
}

/* ══ HERO ══ */
.hero {
  text-align: center;
  padding: 32px 0 36px;
  perspective: 1000px;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(0,212,170,0.08);
  border: 1px solid rgba(0,212,170,0.25);
  border-radius: 999px; padding: 6px 18px;
  font-size: 11px; font-weight: 600; color: var(--teal);
  letter-spacing: .1em; text-transform: uppercase;
  margin-bottom: 20px;
  animation: badge-float 4s ease-in-out infinite;
}
@keyframes badge-float {
  0%,100% { transform: translateY(0px); }
  50%      { transform: translateY(-4px); }
}
.hero h1 {
  font-size: 2.6rem; font-weight: 800; line-height: 1.15;
  margin-bottom: 14px;
  background: linear-gradient(135deg, #e2f0ff 0%, var(--teal) 50%, var(--blue) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: title-shimmer 6s ease-in-out infinite;
  background-size: 200% 200%;
}
@keyframes title-shimmer {
  0%,100% { background-position: 0% 50%; }
  50%      { background-position: 100% 50%; }
}
.hero-sub {
  color: var(--muted); font-size: .92rem; line-height: 1.9;
  max-width: 560px; margin: 0 auto 24px;
}
.hero-stats {
  display: flex; justify-content: center; gap: 32px;
  flex-wrap: wrap; margin-top: 8px;
}
.hero-stat {
  text-align: center;
  animation: stat-appear .6s ease both;
}
.hero-stat-val {
  font-size: 1.5rem; font-weight: 800;
  color: var(--teal);
  text-shadow: 0 0 20px rgba(0,212,170,0.5);
}
.hero-stat-lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }
@keyframes stat-appear {
  from { opacity:0; transform: translateY(10px); }
  to   { opacity:1; transform: translateY(0); }
}

/* ══ 3D CARD WRAPPER ══ */
.card-3d-wrap {
  perspective: 1200px;
  margin-bottom: 20px;
}
.card-3d {
  transform-style: preserve-3d;
  transition: transform .4s ease, box-shadow .4s ease;
}
.card-3d:hover {
  transform: rotateX(2deg) rotateY(-2deg) translateZ(8px);
}

/* ══ GLASS PANELS ══ */
.glass {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 22px 24px;
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  position: relative; overflow: hidden;
  transition: border-color .3s, box-shadow .3s, transform .3s;
  transform-style: preserve-3d;
}
.glass:hover {
  border-color: var(--border2);
  box-shadow: 0 8px 40px rgba(0,212,170,0.12), 0 0 0 1px rgba(0,212,170,0.08);
}
.glass::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
  opacity: .5;
}
.glass::after {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(0,212,170,0.06) 0%, transparent 70%);
  pointer-events: none;
}

/* ══ SECTION LABEL ══ */
.sec-label {
  font-size: 11px; color: var(--muted); letter-spacing: .08em;
  text-transform: uppercase;
  display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
}
.sec-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }
.sec-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ══ UPLOAD ZONE ══ */
[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--border2) !important;
  border-radius: 20px !important;
  background: rgba(0,212,170,0.03) !important;
  backdrop-filter: blur(16px);
  padding: 32px !important;
  transition: all .3s ease;
  position: relative;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--teal) !important;
  background: rgba(0,212,170,0.06) !important;
  box-shadow: 0 0 40px rgba(0,212,170,0.12), inset 0 0 40px rgba(0,212,170,0.03);
}
[data-testid="stFileUploader"] label {
  color: var(--muted) !important;
  font-size: .88rem !important;
}

/* ══ RESULT CARD ══ */
.result-safe   { border-left: 3px solid var(--success) !important; }
.result-danger { border-left: 3px solid var(--danger)  !important; }

.badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; font-weight: 700; letter-spacing: .06em;
  padding: 4px 14px; border-radius: 999px; margin-bottom: 14px;
}
.badge-safe   { background: rgba(0,212,170,.1);  color: var(--success); border: 1px solid rgba(0,212,170,.25); }
.badge-danger { background: rgba(248,113,113,.1); color: var(--danger);  border: 1px solid rgba(248,113,113,.25); }

.risk-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 5px; }
.risk-sub   { font-size: .83rem; color: var(--muted); margin-bottom: 16px; }

/* ══ PROGRESS BAR ══ */
.bar-track {
  height: 9px; border-radius: 999px;
  background: rgba(255,255,255,.05); overflow: hidden; margin: 6px 0 14px;
  box-shadow: inset 0 1px 3px rgba(0,0,0,.3);
}
.bar-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--warn), var(--danger));
  animation: grow-bar 1.4s cubic-bezier(.4,0,.2,1) forwards;
  box-shadow: 0 0 10px rgba(248,113,113,0.5);
}
.bar-fill-safe {
  background: linear-gradient(90deg, #007a62, var(--teal));
  box-shadow: 0 0 10px rgba(0,212,170,0.5);
  animation: grow-bar 1.4s cubic-bezier(.4,0,.2,1) forwards;
}
@keyframes grow-bar { from { width: 0% } }

/* ══ METRIC BOXES ══ */
.metric-row { display: flex; gap: 10px; margin-top: 16px; }
.metric-box {
  flex: 1;
  background: rgba(255,255,255,.03);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 12px 8px; text-align: center;
  transition: border-color .3s, transform .3s;
}
.metric-box:hover {
  border-color: var(--border2);
  transform: translateY(-2px);
}
.metric-val { font-size: 1.1rem; font-weight: 800; }
.metric-lbl { font-size: 10px; color: var(--muted); margin-top: 4px; letter-spacing: .04em; }

/* ══ GRADCAM ══ */
.gradcam-sub { font-size: .82rem; color: var(--muted); line-height: 1.7; margin-bottom: 14px; }
.legend-row  { display: flex; gap: 14px; margin-top: 14px; justify-content: center; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }
.legend-dot  { width: 10px; height: 10px; border-radius: 50%; }

/* ══ DISCLAIMER ══ */
.disclaimer {
  background: rgba(251,191,36,0.05);
  border: 1px solid rgba(251,191,36,0.18);
  border-radius: 14px; padding: 14px 18px;
  font-size: 12px; color: rgba(251,191,36,0.75);
  line-height: 1.7; margin-top: 22px;
}

/* ══ BUTTONS ══ */
.stButton > button {
  border: 1px solid var(--border2) !important;
  background: rgba(0,212,170,0.07) !important;
  color: var(--teal) !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  padding: 11px 24px !important;
  transition: all .25s ease !important;
  width: 100%;
  letter-spacing: .04em;
}
.stButton > button:hover {
  background: rgba(0,212,170,0.16) !important;
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,212,170,0.22) !important;
  border-color: var(--teal) !important;
}
.stDownloadButton > button {
  width: 100% !important;
  border: 1.5px solid var(--border2) !important;
  background: rgba(0,212,170,0.08) !important;
  color: var(--teal) !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  font-size: .95rem !important;
  padding: 14px 24px !important;
  letter-spacing: .04em;
  transition: all .25s ease !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,212,170,0.18) !important;
  border-color: var(--teal) !important;
  box-shadow: 0 8px 28px rgba(0,212,170,0.25) !important;
  transform: translateY(-2px);
}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
  background: rgba(2,14,31,0.92) !important;
  backdrop-filter: blur(24px);
  border-right: 1px solid var(--border) !important;
}

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: var(--teal) !important; }

/* ══ MISC ══ */
footer, #MainMenu, header { visibility: hidden !important; }
h1,h2,h3,h4,h5,h6,p,span,label { color: var(--txt); }
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--teal3), var(--teal)) !important;
  border-radius: 999px !important;
}

/* ══ SCAN LINE ANIMATION ══ */
.scan-line {
  position: absolute; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
  animation: scan 3s linear infinite;
  opacity: 0.6;
  pointer-events: none;
}
@keyframes scan {
  0%   { top: 0%; opacity: 0; }
  5%   { opacity: 0.6; }
  95%  { opacity: 0.6; }
  100% { top: 100%; opacity: 0; }
}

/* ══ DNA HELIX DECORATION ══ */
.dna-wrap {
  position: fixed; right: -30px; top: 50%;
  transform: translateY(-50%);
  pointer-events: none; z-index: 0;
  opacity: 0.07;
}

/* ══ FEATURE PILLS ══ */
.feature-pills {
  display: flex; flex-wrap: wrap; gap: 8px;
  justify-content: center; margin-top: 18px;
}
.feature-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(56,189,248,0.07);
  border: 1px solid rgba(56,189,248,0.18);
  border-radius: 999px; padding: 5px 14px;
  font-size: 11.5px; color: var(--blue);
  font-weight: 500;
}

/* ══ STEP INDICATOR ══ */
.steps-row {
  display: flex; align-items: center; justify-content: center;
  gap: 0; margin: 24px 0 28px; flex-wrap: wrap;
}
.step-item {
  display: flex; flex-direction: column; align-items: center;
  gap: 6px; min-width: 90px;
}
.step-circle {
  width: 38px; height: 38px; border-radius: 50%;
  background: rgba(0,212,170,0.08);
  border: 1.5px solid var(--border2);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  transition: all .3s;
}
.step-circle.active {
  background: rgba(0,212,170,0.18);
  border-color: var(--teal);
  box-shadow: 0 0 16px rgba(0,212,170,0.35);
}
.step-lbl { font-size: 10px; color: var(--muted); text-align: center; max-width: 80px; }
.step-connector {
  flex: 1; height: 1px; min-width: 20px;
  background: linear-gradient(90deg, var(--border2), var(--border));
  margin-bottom: 22px;
}

/* ══ IMAGE PREVIEW FRAME ══ */
.img-frame {
  border-radius: 16px; overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,212,170,0.08);
  position: relative;
}
</style>

<!-- ══ BACKGROUND LAYERS (CSS-only, works everywhere) ══ -->

<!-- Glowing orbs -->
<div class="orb-layer">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
  <div class="orb orb-4"></div>
  <div class="orb orb-5"></div>
</div>

<!-- Grid -->
<div class="grid-overlay"></div>

<!-- Rotating conic sweep -->
<div class="sweep-layer"></div>

<!-- Center pulse glow -->
<div class="center-glow"></div>

<!-- Spinning ellipse rings at corners -->
<div class="rings-layer">
  <div class="ring ring-tl-1"></div>
  <div class="ring ring-tl-2"></div>
  <div class="ring ring-tl-3"></div>
  <div class="ring ring-br-1"></div>
  <div class="ring ring-br-2"></div>
  <div class="ring ring-br-3"></div>
  <div class="ring ring-tr-1"></div>
  <div class="ring ring-tr-2"></div>
  <div class="ring ring-bl-1"></div>
  <div class="ring ring-bl-2"></div>
</div>

<!-- Floating particles (CSS animated) -->
<div class="particles-layer">
  <div class="p-dot" style="width:4px;height:4px;background:#00d4aa;left:8%;animation-duration:14s;animation-delay:0s;box-shadow:0 0 8px #00d4aa;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#38bdf8;left:15%;animation-duration:18s;animation-delay:-3s;box-shadow:0 0 6px #38bdf8;"></div>
  <div class="p-dot" style="width:5px;height:5px;background:#00d4aa;left:23%;animation-duration:12s;animation-delay:-6s;box-shadow:0 0 10px #00d4aa;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#a78bfa;left:31%;animation-duration:20s;animation-delay:-1s;box-shadow:0 0 6px #a78bfa;"></div>
  <div class="p-dot" style="width:4px;height:4px;background:#38bdf8;left:40%;animation-duration:16s;animation-delay:-8s;box-shadow:0 0 8px #38bdf8;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#00d4aa;left:48%;animation-duration:11s;animation-delay:-4s;box-shadow:0 0 6px #00d4aa;"></div>
  <div class="p-dot" style="width:5px;height:5px;background:#38bdf8;left:56%;animation-duration:22s;animation-delay:-10s;box-shadow:0 0 10px #38bdf8;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#00d4aa;left:63%;animation-duration:15s;animation-delay:-2s;box-shadow:0 0 6px #00d4aa;"></div>
  <div class="p-dot" style="width:4px;height:4px;background:#a78bfa;left:71%;animation-duration:19s;animation-delay:-7s;box-shadow:0 0 8px #a78bfa;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#00d4aa;left:79%;animation-duration:13s;animation-delay:-5s;box-shadow:0 0 6px #00d4aa;"></div>
  <div class="p-dot" style="width:5px;height:5px;background:#38bdf8;left:87%;animation-duration:17s;animation-delay:-9s;box-shadow:0 0 10px #38bdf8;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#00d4aa;left:93%;animation-duration:21s;animation-delay:-11s;box-shadow:0 0 6px #00d4aa;"></div>
  <div class="p-dot" style="width:4px;height:4px;background:#a78bfa;left:4%;animation-duration:16s;animation-delay:-13s;box-shadow:0 0 8px #a78bfa;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#38bdf8;left:35%;animation-duration:24s;animation-delay:-15s;box-shadow:0 0 6px #38bdf8;"></div>
  <div class="p-dot" style="width:5px;height:5px;background:#00d4aa;left:52%;animation-duration:10s;animation-delay:-12s;box-shadow:0 0 10px #00d4aa;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#a78bfa;left:68%;animation-duration:23s;animation-delay:-16s;box-shadow:0 0 6px #a78bfa;"></div>
  <div class="p-dot" style="width:4px;height:4px;background:#38bdf8;left:82%;animation-duration:14s;animation-delay:-18s;box-shadow:0 0 8px #38bdf8;"></div>
  <div class="p-dot" style="width:3px;height:3px;background:#00d4aa;left:96%;animation-duration:19s;animation-delay:-20s;box-shadow:0 0 6px #00d4aa;"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def img_to_b64(image_pil: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    image_pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🔬</div>
    <div>
      <div class="brand-name">Artificial intelligence of the skin(DermAI)</div>
      <div class="brand-ver">v3.0 · EfficientNet</div>
    </div>
  </div>
  <div class="status-pill"><span class="s-dot"></span>سیستم آماده</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🧬 هوش مصنوعی پزشکی · تشخیص سرطان پوست</div>
  <h1>سامانه تشخیص ملانوما</h1>
  <p class="hero-sub">          
    با استفاده از شبکه عصبی عمیق EfficientNet، تصویر ضایعه پوستی را آپلود کنید
    و در چند ثانیه نتیجه تحلیل هوش مصنوعی را دریافت نمایید.       
  </p>
  <div class="feature-pills">
    <span class="feature-pill">⚡ EfficientNet B3</span>
    <span class="feature-pill">🗺️ Grad-CAM</span>
    <span class="feature-pill">📄 گزارش PDF</span>
    <span class="feature-pill">🎯 دقت بالا</span>
  </div>
  <div class="hero-stats">
    <div class="hero-stat" style="animation-delay:.1s">
      <div class="hero-stat-val">224×224</div>
      <div class="hero-stat-lbl">رزولوشن ورودی</div>
    </div>
    <div class="hero-stat" style="animation-delay:.2s">
      <div class="hero-stat-val">EfficientNet</div>
      <div class="hero-stat-lbl">معماری مدل</div>
    </div>
    <div class="hero-stat" style="animation-delay:.3s">
      <div class="hero-stat-val">Grad-CAM</div>
      <div class="hero-stat-lbl">تفسیرپذیری</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STEP INDICATOR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="steps-row">
  <div class="step-item">
    <div class="step-circle active">📤</div>
    <div class="step-lbl">آپلود تصویر</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">🧠</div>
    <div class="step-lbl">تحلیل AI</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">🗺️</div>
    <div class="step-lbl">Grad-CAM</div>
  </div>
  <div class="step-connector"></div>
  <div class="step-item">
    <div class="step-circle">📄</div>
    <div class="step-lbl">گزارش PDF</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(path):
    return tf.keras.models.load_model(path, compile=False)

model = None
with st.spinner("⚡ بارگذاری مدل هوش مصنوعی..."):
    try:
        model = load_model_cached(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ خطا در بارگذاری مدل: {e}", icon=":material/error:")


# ─────────────────────────────────────────────────────────────
# PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr   = tf.keras.preprocessing.image.img_to_array(image)
    arr   = np.expand_dims(arr, 0)
    return tf.keras.applications.efficientnet.preprocess_input(arr)


def predict_probs(model, arr):
    pred   = float(model.predict(arr, verbose=0)[0][0])
    benign = np.clip(pred, 0.0, 1.0)
    return benign, 1.0 - benign


def make_gradcam(img_array, model):

    base_model = model.layers[0]

    # لایه مورد نظر
    last_conv_layer_name = "block6f_project_conv"

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, base_output = grad_model(img_array)

        x = base_output

        # ادامه مسیر مدل بعد از EfficientNet
        for layer in model.layers[1:]:
            x = layer(x)

        loss = x[:,0]


    grads = tape.gradient(loss, conv_outputs)


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)


    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.reduce_max(heatmap)


    return heatmap.numpy(), last_conv_layer_name


def overlay_heatmap(image_pil, heatmap, alpha=0.45):
    rgb  = np.array(image_pil.convert("RGB"))
    bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h    = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))
    h    = cv2.GaussianBlur(h, (5, 5), 0)
    h_u8 = np.uint8(255 * h)
    h_col = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(bgr, 1 - alpha, h_col, alpha, 0)
    return cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────────────────────────
def build_pdf(orig_img: Image.Image, cam_img_arr,
              melanoma_prob: float, benign_prob: float,
              suspicious: bool, layer_name: str = "") -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Image as RLImage, Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import tempfile, os

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 rightMargin=2*cm, leftMargin=2*cm,
                                 topMargin=2*cm,   bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        TEAL   = colors.HexColor("#00d4aa")
        DARK   = colors.HexColor("#020b18")
        DANGER = colors.HexColor("#f87171")
        SAFE   = colors.HexColor("#00d4aa")
        MUTED  = colors.HexColor("#5a8aaa")

        title_style  = ParagraphStyle("title",  fontName="Helvetica-Bold", fontSize=20,
                                       textColor=TEAL,  spaceAfter=4,  alignment=1)
        sub_style    = ParagraphStyle("sub",    fontName="Helvetica",      fontSize=10,
                                       textColor=MUTED, spaceAfter=2,  alignment=1)
        label_style  = ParagraphStyle("label",  fontName="Helvetica-Bold", fontSize=11,
                                       textColor=DARK,  spaceAfter=2)
        body_style   = ParagraphStyle("body",   fontName="Helvetica",      fontSize=10,
                                       textColor=DARK,  spaceAfter=4)

        def pil_to_tmp(img_pil, suffix=".jpg"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            img_pil.save(tmp.name, "JPEG", quality=90)
            return tmp.name

        orig_sq = orig_img.convert("RGB")
        orig_sq.thumbnail((300, 300))
        orig_path = pil_to_tmp(orig_sq)

        story = []
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("DermAI — Melanoma Detection Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}", sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=12))

        result_color = DANGER if suspicious else SAFE
        result_text  = "HIGH RISK – Suspicious for Melanoma" if suspicious else "LOW RISK – Likely Benign"
        banner_style = ParagraphStyle("banner", fontName="Helvetica-Bold", fontSize=14,
                                       textColor=result_color, spaceAfter=6, alignment=1)
        story.append(Paragraph(result_text, banner_style))
        story.append(Spacer(1, 0.3*cm))

        pct_mel    = f"{melanoma_prob*100:.2f}%"
        pct_ben    = f"{benign_prob*100:.2f}%"
        risk_label = "High" if suspicious else "Low"

        tdata = [
            [Paragraph("<b>Metric</b>", label_style), Paragraph("<b>Value</b>", label_style)],
            ["Melanoma Probability", pct_mel],
            ["Benign Probability",   pct_ben],
            ["Risk Level",           risk_label],
            ["Threshold Used",       f"{THRESHOLD*100:.0f}%"],
        ]
        if layer_name:
            tdata.append(["Grad-CAM Layer", layer_name])

        tbl = Table(tdata, colWidths=[9*cm, 7*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#041428")),
            ("TEXTCOLOR",     (0,0), (-1,0), TEAL),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f9f9f7"), colors.white]),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#d1d5db")),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5*cm))

        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#d1d5db"), spaceAfter=10))
        story.append(Paragraph("Original Image", label_style))
        story.append(RLImage(orig_path, width=7*cm, height=7*cm))
        story.append(Spacer(1, 0.4*cm))

        if cam_img_arr is not None:
            cam_pil  = Image.fromarray(cam_img_arr.astype(np.uint8))
            cam_pil.thumbnail((300, 300))
            cam_path = pil_to_tmp(cam_pil)
            story.append(Paragraph("Grad-CAM Heatmap", label_style))
            story.append(Paragraph(
                "Warmer areas (red/orange) indicate regions of higher model attention.",
                body_style))
            story.append(RLImage(cam_path, width=7*cm, height=7*cm))
            story.append(Spacer(1, 0.4*cm))

        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=TEAL, spaceBefore=10, spaceAfter=8))
        disc_style = ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=9,
                                     textColor=MUTED, spaceAfter=0)
        story.append(Paragraph(
            "⚠ This report is generated by an AI-assisted tool and is intended for research "
            "and supplementary clinical use only. It must not be used as the sole basis for a "
            "clinical diagnosis. Please consult a qualified dermatologist for any medical decisions.",
            disc_style))

        doc.build(story)
        os.unlink(orig_path)
        if cam_img_arr is not None:
            try: os.unlink(cam_path)
            except: pass
        return buf.getvalue()

    except ImportError:
        buf = io.BytesIO()
        buf.write(b"%PDF-1.4\n% reportlab not installed\n")
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass" style="margin-bottom:20px;">
  <div class="sec-label">
    <span class="sec-dot"></span>
    آپلود تصویر ضایعه پوستی
    <span class="sec-line"></span>
  </div>
  <p style="font-size:.85rem; color:var(--muted); margin-bottom:14px; line-height:1.7;">
    تصویر واضح و با کیفیت از ضایعه پوستی آپلود کنید. فرمت‌های JPG، JPEG و PNG پشتیبانی می‌شوند.
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], key="main_img")


# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────
if uploaded_file and model:
    image     = Image.open(uploaded_file)
    processed = preprocess_image(image)

    col_img, col_res = st.columns(2, gap="medium")

    with col_img:
        st.markdown("""
        <div class="glass">
          <div class="sec-label">
            <span class="sec-dot"></span>تصویر ورودی<span class="sec-line"></span>
          </div>
          <div class="scan-line"></div>
        </div>""", unsafe_allow_html=True)
        st.image(image, width="stretch")

    with col_res:
        with st.spinner("🧠 در حال تحلیل تصویر با هوش مصنوعی..."):
            time.sleep(0.15)
            benign_prob, melanoma_prob = predict_probs(model, processed)
            suspicious = melanoma_prob >= THRESHOLD

        badge_cls  = "badge-danger" if suspicious else "badge-safe"
        result_cls = "result-danger" if suspicious else "result-safe"
        badge_txt  = "⬤ ریسک بالا"          if suspicious else "⬤ ریسک پایین"
        title      = "🚨 مشکوک به ملانوما"   if suspicious else "✅ خوش‌خیم / طبیعی"
        title_col  = "var(--danger)"          if suspicious else "var(--success)"
        bar_pct    = melanoma_prob * 100
        bar_class  = "bar-fill"               if suspicious else "bar-fill bar-fill-safe"
        risk_lbl   = "بالا"                   if suspicious else "پایین"
        risk_col   = "var(--danger)"          if suspicious else "var(--success)"

        st.markdown(f"""
        <div class="glass {result_cls}">
          <div class="sec-label">
            <span class="sec-dot"></span>نتیجه تحلیل<span class="sec-line"></span>
          </div>
          <span class="badge {badge_cls}">{badge_txt}</span>
          <div class="risk-title" style="color:{title_col}">{title}</div>
          <div class="risk-sub">احتمال ملانوما: <b>{melanoma_prob*100:.2f}٪</b></div>
          <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:4px">
            <span>نوار ریسک</span>
            <b style="color:{title_col}">{melanoma_prob*100:.1f}٪</b>
          </div>
          <div class="bar-track">
            <div class="{bar_class}" style="width:{bar_pct:.1f}%"></div>
          </div>
          <div class="metric-row">
            <div class="metric-box">
              <div class="metric-val" style="color:var(--danger)">{melanoma_prob*100:.1f}٪</div>
              <div class="metric-lbl">ملانوما</div>
            </div>
            <div class="metric-box">
              <div class="metric-val" style="color:var(--success)">{benign_prob*100:.1f}٪</div>
              <div class="metric-lbl">خوش‌خیم</div>
            </div>
            <div class="metric-box">
              <div class="metric-val" style="color:{risk_col}">{risk_lbl}</div>
              <div class="metric-lbl">ریسک</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── GRAD-CAM ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
      <div class="sec-label">
        <span class="sec-dot"></span>نقشه حرارتی Grad-CAM<span class="sec-line"></span>
      </div>
      <div class="gradcam-sub">
        نقشه Grad-CAM نشان می‌دهد مدل هوش مصنوعی به کدام نواحی تصویر بیشتر توجه کرده است.
        نواحی قرمز/نارنجی نشان‌دهنده بیشترین توجه مدل هستند.
      </div>
    </div>""", unsafe_allow_html=True)

    cam_result = None
    layer_used = ""

    if st.button("🗺️ تولید نقشه حرارتی Grad-CAM", key="gradcam_btn"):
        with st.spinner("⚙️ محاسبه گرادیان‌ها و تولید نقشه حرارتی..."):
            try:
                heatmap, layer_used = make_gradcam(processed, model)
                cam_result = overlay_heatmap(image, heatmap, alpha=0.45)
                st.success(f"✅ نقشه حرارتی تولید شد · لایه: `{layer_used}`",
                           icon=":material/check_circle:")
                st.image(cam_result, caption="Grad-CAM Visualization")
                st.markdown("""
                <div class="legend-row">
                  <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>توجه کم</div>
                  <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>توجه متوسط</div>
                  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>توجه زیاد</div>
                  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>توجه بحرانی</div>
                </div>""", unsafe_allow_html=True)
                st.session_state["cam_result"] = cam_result
                st.session_state["layer_used"] = layer_used
            except Exception as e:
                st.error(f"❌ خطا در تولید Grad-CAM: {e}", icon=":material/error:")

    if "cam_result" in st.session_state and cam_result is None:
        cam_result = st.session_state["cam_result"]
        layer_used = st.session_state.get("layer_used", "")

    # ── PDF DOWNLOAD ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("📄 آماده‌سازی گزارش PDF..."):
        pdf_bytes = build_pdf(
            orig_img      = image,
            cam_img_arr   = cam_result,
            melanoma_prob = melanoma_prob,
            benign_prob   = benign_prob,
            suspicious    = suspicious,
            layer_name    = layer_used,
        )

    fname = f"DermAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    st.download_button(
        label     = "📄 دانلود گزارش کامل PDF",
        data      = pdf_bytes,
        file_name = fname,
        mime      = "application/pdf",
    )

    # ── DISCLAIMER ──
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <b>هشدار پزشکی:</b> این سامانه صرفاً یک ابزار کمکی مبتنی بر هوش مصنوعی است و
      نباید جایگزین تشخیص کلینیکی توسط متخصص پوست و مو شود. در صورت مشاهده هرگونه
      ضایعه مشکوک، حتماً با پزشک متخصص مشورت نمایید.
    </div>""", unsafe_allow_html=True)

else:
    # ── EMPTY STATE ──
    st.markdown("""
    <div class="glass" style="text-align:center; padding: 48px 24px; margin-top: 8px;">
      <div style="font-size: 3.5rem; margin-bottom: 16px; animation: badge-float 3s ease-in-out infinite;">🔬</div>
      <div style="font-size: 1.1rem; font-weight: 700; color: var(--txt); margin-bottom: 8px;">
        آماده تحلیل تصویر پوستی
      </div>
      <div style="font-size: .85rem; color: var(--muted); line-height: 1.8; max-width: 380px; margin: 0 auto;">
        برای شروع، یک تصویر از ضایعه پوستی را در کادر بالا آپلود کنید.
        سیستم هوش مصنوعی در چند ثانیه نتیجه را نمایش می‌دهد.
      </div>
      <div style="margin-top: 24px; display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
        <div style="background: rgba(0,212,170,0.06); border: 1px solid var(--border); border-radius: 12px; padding: 12px 20px; font-size: 12px; color: var(--muted);">
          🖼️ JPG / JPEG / PNG
        </div>
        <div style="background: rgba(0,212,170,0.06); border: 1px solid var(--border); border-radius: 12px; padding: 12px 20px; font-size: 12px; color: var(--muted);">
          📐 حداقل 224×224 پیکسل
        </div>
        <div style="background: rgba(0,212,170,0.06); border: 1px solid var(--border); border-radius: 12px; padding: 12px 20px; font-size: 12px; color: var(--muted);">
          ⚡ نتیجه در چند ثانیه
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR – INFO
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px;">
      <div style="font-size: 1rem; font-weight: 700; color: var(--teal); margin-bottom: 4px;">🔬 DermAI</div>
      <div style="font-size: 11px; color: var(--muted);">سامانه تشخیص ملانوما</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**درباره سامانه**")
    st.caption(
        "این سامانه از شبکه عصبی عمیق EfficientNet برای تشخیص ملانوما "
        "از روی تصاویر ضایعات پوستی استفاده می‌کند."
    )
    st.markdown("---")
    st.markdown("**راهنمای استفاده**")
    st.caption("۱. تصویر واضح از ضایعه پوستی آپلود کنید")
    st.caption("۲. منتظر تحلیل هوش مصنوعی بمانید")
    st.caption("۳. نقشه Grad-CAM را تولید کنید")
    st.caption("۴. گزارش PDF دانلود کنید")
    st.markdown("---")
    st.caption("⚠️ این ابزار جایگزین تشخیص پزشکی نیست.")
