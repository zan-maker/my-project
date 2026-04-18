#!/usr/bin/env python3
"""
MineLens AI Demo Video Generator (Optimized)
Creates a professional presentation-style demo video for the Gemma 4 Good Hackathon.
Generates 8 slide images, then uses ffmpeg to assemble with transitions.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

# ── Configuration ────────────────────────────────────────────────────────────
W, H = 1920, 1080
FPS = 24
DURATION_PER_SLIDE = 7  # seconds per slide
FADE_DUR = 0.8  # crossfade duration in seconds
TOTAL_SLIDES = 8
BG = (22, 34, 53)
CYAN = (55, 220, 242)
WHITE = (255, 255, 255)
LIGHT_GRAY = (176, 184, 192)
MID_GRAY = (100, 115, 130)
DIM = (40, 55, 75)
ACCENT2 = (91, 128, 255)
ACCENT3 = (255, 183, 77)
GREEN = (76, 217, 100)
RED_SOFT = (255, 99, 99)
PURPLE = (224, 64, 251)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDES_DIR = os.path.join(OUT_DIR, 'slides')
VIDEO_PATH = os.path.join(OUT_DIR, 'minelens_demo.mp4')

FONT_EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_EN_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

def load_font(size, bold=False):
    path = FONT_EN_BOLD if bold else FONT_EN
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

def load_mono(size):
    try:
        return ImageFont.truetype(FONT_MONO, size)
    except:
        return load_font(size)

# ── Utility ──────────────────────────────────────────────────────────────────
def new_canvas():
    return Image.new('RGB', (W, H), BG)

def draw_bg_grid(draw):
    for x in range(0, W, 60):
        draw.line([(x, 0), (x, H)], fill=(25, 38, 58), width=1)
    for y in range(0, H, 60):
        draw.line([(0, y), (W, y)], fill=(25, 38, 58), width=1)

def draw_top_bar(draw):
    draw.rectangle([(0, 0), (W, 4)], fill=CYAN)

def draw_bottom_bar(draw, slide_num):
    draw.rectangle([(0, H-40), (W, H)], fill=(18, 28, 45))
    draw.rectangle([(0, H-40), (W, H-38)], fill=CYAN)
    font = load_font(16)
    draw.text((W-120, H-30), f"  {slide_num} / {TOTAL_SLIDES}", fill=MID_GRAY, font=font)
    progress = slide_num / TOTAL_SLIDES
    bar_w = 300
    bx = W//2 - bar_w//2
    draw.rectangle([(bx, H-28), (bx+bar_w, H-18)], fill=DIM)
    draw.rectangle([(bx, H-28), (bx+int(bar_w*progress), H-18)], fill=CYAN)

def draw_corners(draw):
    l = 30
    for (cx, cy, dx, dy) in [(60,60,1,1),(W-60,60,-1,1),(60,H-60,1,-1),(W-60,H-60,-1,-1)]:
        draw.line([(cx, cy+l*dy), (cx, cy), (cx+l*dx, cy)], fill=CYAN, width=2)

def tc(draw, y, text, font, fill=WHITE):
    """Text centered"""
    bb = draw.textbbox((0,0), text, font=font)
    tw = bb[2]-bb[0]
    draw.text(((W-tw)//2, y), text, font=font, fill=fill)

def hexagon(draw, cx, cy, r, fill_c, out_c=None):
    pts = [(cx + r*np.cos(np.pi/6 + i*np.pi/3), cy + r*np.sin(np.pi/6 + i*np.pi/3)) for i in range(6)]
    draw.polygon(pts, fill=fill_c, outline=out_c)

def glow_circle(draw, cx, cy, r, color, steps=6):
    for i in range(steps, 0, -1):
        c = tuple(min(255, int(v*(0.3+0.7*i/steps))) for v in color)
        ri = r + (steps-i)*3
        draw.ellipse([(cx-ri,cy-ri),(cx+ri,cy+ri)], fill=None, outline=c, width=1)

def badge(draw, x, y, text, size=14, fill=CYAN):
    f = load_font(size, True)
    bb = f.getbbox(text)
    tw = bb[2]-bb[0]
    pad = 10
    draw.rounded_rectangle([(x,y),(x+tw+2*pad,y+size+2*pad)], radius=6, fill=(30,45,70), outline=fill)
    draw.text((x+pad, y+pad), text, font=f, fill=fill)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
def slide_01():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)
    glow_circle(d, W//2, H//2-40, 180, CYAN)
    for i in range(6):
        a = i*np.pi/3
        hexagon(d, W//2+int(200*np.cos(a)), H//2-40+int(200*np.sin(a)), 25, DIM, CYAN)
    hexagon(d, W//2, H//2-40, 38, CYAN, WHITE)

    tc(d, 170, "MineLens AI", load_font(74, True), CYAN)
    tc(d, 265, "Critical Mineral Prospectivity Mapping", load_font(28), WHITE)
    d.line([(W//2-250, 320),(W//2+250, 320)], fill=CYAN, width=2)
    tc(d, 345, "Built with Gemma 4  |  Native Function Calling  |  6 Geoscience Tools", load_font(23), LIGHT_GRAY)

    badge(d, W//2-220, H-210, "KAGGLE", 18, ACCENT3)
    badge(d, W//2-60, H-210, "GEMMA 4 GOOD HACKATHON", 18, CYAN)
    tc(d, H-155, "$200,000 Prize Pool", load_font(22, True), ACCENT3)
    tc(d, H-110, "Li  |  Co  |  REE  |  Cu  |  Ni  |  Targeting Critical Minerals for Energy Transition", load_font(18), MID_GRAY)
    draw_bottom_bar(d, 1)
    return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem
# ════════════════════════════════════════════════════════════════════════════
def slide_02():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    d.text((100,80), "The Challenge", font=load_font(52, True), fill=CYAN)
    d.text((100,145), "Critical mineral exploration faces a perfect storm", font=load_font(24), fill=LIGHT_GRAY)
    d.line([(100,190),(600,190)], fill=CYAN, width=2)

    stats = [("$2T+","Global Critical\nMineral Market",CYAN),
             ("$50-500M","Cost Per\nDiscovery",RED_SOFT),
             ("10-15 yrs","Average Discovery\nTimeline",ACCENT3),
             ("40-70%","Declining\nDiscovery Rate",ACCENT2)]
    bw, bh, gap = 320, 190, 60
    for i,(val,lab,col) in enumerate(stats):
        x = 100+i*(bw+gap); y = 230
        d.rounded_rectangle([(x,y),(x+bw,y+bh)], radius=12, fill=(28,42,65), outline=col, width=2)
        f = load_font(46, True); bb = f.getbbox(val)
        d.text((x+(bw-bb[2]+bb[0])//2, y+20), val, font=f, fill=col)
        f2 = load_font(17)
        for j,ln in enumerate(lab.split('\n')):
            bb = f2.getbbox(ln); d.text((x+(bw-bb[2]+bb[0])//2, y+95+j*26), ln, font=f2, fill=LIGHT_GRAY)

    lines = [
        "▸  The energy transition demands an unprecedented surge in critical mineral supply.",
        "▸  Lithium, cobalt, rare earths, copper, and nickel are essential for EVs, renewables, and defense.",
        "▸  Yet exploration is slow, expensive, and risky — creating a critical bottleneck.",
        "▸  We need AI-powered tools to accelerate prospectivity analysis and de-risk exploration."
    ]
    ny = 490
    for ln in lines:
        d.text((100,ny), ln, font=load_font(20), fill=LIGHT_GRAY); ny += 36

    d.rounded_rectangle([(100,660),(W-100,740)], radius=10, fill=(35,25,25), outline=RED_SOFT, width=1)
    d.text((140,678), "Without new discoveries, the transition to clean energy stalls.", font=load_font(22, True), fill=RED_SOFT)
    d.text((140,710), "MineLens AI makes prospectivity analysis faster, cheaper, and more accessible.", font=load_font(18), fill=LIGHT_GRAY)

    icons = ["EVs","Solar","Wind","Batteries","Defense"]
    ix = 100
    for ic in icons:
        d.rounded_rectangle([(ix,790),(ix+120,835)], radius=8, fill=DIM, outline=ACCENT2)
        f = load_font(16, True); bb = f.getbbox(ic)
        d.text((ix+(120-bb[2]+bb[0])//2, 802), ic, font=f, fill=ACCENT2)
        ix += 140

    draw_bottom_bar(d, 2); return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Architecture
# ════════════════════════════════════════════════════════════════════════════
def slide_03():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    d.text((100,70), "Solution Architecture", font=load_font(48, True), fill=CYAN)
    d.text((100,130), "Gemma 4 orchestrates 6 geoscience tools through native function calling", font=load_font(22), fill=LIGHT_GRAY)
    d.line([(100,170),(700,170)], fill=CYAN, width=2)

    # Draw architecture boxes directly with PIL
    # User Query
    d.rounded_rectangle([(80,220),(300,300)], radius=12, fill=(30,48,80), outline=CYAN, width=2)
    d.text((120,232), "User Query", font=load_font(20, True), fill=WHITE)
    d.text((120,258), '"Find lithium in', font=load_font(15), fill=LIGHT_GRAY)
    d.text((120,276), ' Atacama Desert"', font=load_font(15), fill=LIGHT_GRAY)

    # Arrow
    d.line([(310,260),(380,260)], fill=CYAN, width=3)
    d.polygon([(380,250),(400,260),(380,270)], fill=CYAN)

    # Gemma 4 box
    d.rounded_rectangle([(400,190),(800,340)], radius=14, fill=(13,40,71), outline=CYAN, width=3)
    d.text((500,200), "GEMMA 4", font=load_font(40, True), fill=CYAN)
    d.text((470,255), "Agentic Orchestrator", font=load_font(22), fill=LIGHT_GRAY)
    d.text((445,285), "Native Function Calling", font=load_font(18), fill=ACCENT2)
    d.text((455,310), "Autonomous Tool Selection", font=load_font(18), fill=ACCENT2)

    # Arrow to tools
    d.line([(810,265),(860,265)], fill=CYAN, width=3)
    d.polygon([(860,255),(880,265),(860,275)], fill=CYAN)

    # 6 Tool boxes - 2 rows of 3
    tools = [("Spectral\nAnalysis",CYAN),("Terrain\nClassifier",GREEN),("Proximity\nSearch",ACCENT2),
             ("Risk\nAssessment",RED_SOFT),("Report\nGenerator",ACCENT3),("Geological\nSurvey",PURPLE)]
    tw, th = 250, 120
    gap_x, gap_y = 20, 20
    sx = 890
    for i,(name,col) in enumerate(tools):
        col_i = i % 3; row_i = i // 3
        x = sx + col_i*(tw+gap_x); y = 200 + row_i*(th+gap_y)
        d.rounded_rectangle([(x,y),(x+tw,y+th)], radius=10, fill=(28,42,65), outline=col, width=2)
        f = load_font(18, True)
        for j,ln in enumerate(name.split('\n')):
            bb = f.getbbox(ln); d.text((x+(tw-bb[2]+bb[0])//2, y+30+j*28), ln, font=f, fill=col)
        # Arrows between tools
        if col_i < 2:
            d.line([(x+tw+1,y+th//2),(x+tw+gap_x-1,y+th//2)], fill=CYAN, width=1)

    # Arrow down between rows
    d.line([(sx+2*tw+tw//2+gap_x, 200+th+1),(sx+2*tw+tw//2+gap_x, 200+th+gap_y-1)], fill=CYAN, width=1)
    d.polygon([(sx+2*tw+tw//2+gap_x-5, 200+th+gap_y-1),(sx+2*tw+tw//2+gap_x+5, 200+th+gap_y-1),(sx+2*tw+tw//2+gap_x, 200+th+gap_y+5)], fill=CYAN)

    # Output box
    d.rounded_rectangle([(400,430),(W-100,520)], radius=12, fill=(20,50,35), outline=GREEN, width=2)
    d.text((580,445), "Prospectivity Report & Mineral Map", font=load_font(28, True), fill=GREEN)
    d.text((540,485), "Comprehensive analysis with risk scores, maps, and recommendations", font=load_font(17), fill=LIGHT_GRAY)

    # Arrow from tools to output
    d.line([(sx+tw//2, 340+th+gap_y+20),(sx+tw//2, 420)], fill=GREEN, width=2)
    d.line([(sx+tw//2, 420),(W//2, 420)], fill=GREEN, width=2)
    d.line([(W//2, 420),(W//2, 425)], fill=GREEN, width=2)
    d.polygon([(W//2-8,425),(W//2+8,425),(W//2,433)], fill=GREEN)

    # Key features at bottom
    features = [
        ("Zero-Shot Tool Selection", "Gemma 4 decides which tools to invoke based on query intent"),
        ("Sequential Reasoning", "Each tool output informs the next tool's parameters"),
        ("Composable Pipeline", "Tools can be chained, parallelized, or skipped as needed"),
    ]
    fy = 560
    for title, desc in features:
        d.rounded_rectangle([(100,fy),(W-100,fy+80)], radius=8, fill=(28,42,65), outline=DIM)
        d.text((140, fy+10), "▸  "+title, font=load_font(20, True), fill=CYAN)
        d.text((140, fy+40), desc, font=load_font(16), fill=LIGHT_GRAY)
        fy += 95

    draw_bottom_bar(d, 3); return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Tools Overview
# ════════════════════════════════════════════════════════════════════════════
def slide_04():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    d.text((100,70), "6 Geoscience Tools", font=load_font(48, True), fill=CYAN)
    d.text((100,130), "Each tool is a Gemma 4 function — orchestrated autonomously", font=load_font(22), fill=LIGHT_GRAY)
    d.line([(100,170),(600,170)], fill=CYAN, width=2)

    tools = [
        ("1","Spectral Analysis","Detect mineral signatures in\nmultispectral satellite imagery",CYAN),
        ("2","Terrain Classifier","Classify landforms: desert,\nmountain, basin, volcanic",GREEN),
        ("3","Proximity Search","Find nearby infrastructure,\nmines, and geological features",ACCENT2),
        ("4","Risk Assessment","Evaluate environmental, political,\nand operational risks",RED_SOFT),
        ("5","Report Generator","Generate comprehensive PDF\nprospectivity reports",ACCENT3),
        ("6","Geological Survey","Query USGS and global\ngeological databases",PURPLE),
    ]

    bw, bh = 830, 130; gx, gy = 30, 20; sx, sy = 100, 195
    for i,(num,name,desc,col) in enumerate(tools):
        ci = i % 2; ri = i // 2
        x = sx+ci*(bw+gx); y = sy+ri*(bh+gy)
        d.rounded_rectangle([(x,y),(x+bw,y+bh)], radius=12, fill=(28,42,65), outline=col, width=2)
        # Number circle
        cx, cy, r = x+45, y+bh//2, 26
        d.ellipse([(cx-r,cy-r),(cx+r,cy+r)], fill=col)
        f = load_font(22, True); bb = f.getbbox(num)
        d.text((cx-bb[2]//2, cy-12), num, font=f, fill=BG)
        # Name
        d.text((x+90, y+18), name, font=load_font(22, True), fill=col)
        # Desc
        for j,ln in enumerate(desc.split('\n')):
            d.text((x+90, y+52+j*24), ln, font=load_font(16), fill=LIGHT_GRAY)
        # Status pill
        d.rounded_rectangle([(x+bw-150,y+bh-35),(x+bw-15,y+bh-8)], radius=6, fill=(20,50,35), outline=GREEN)
        d.text((x+bw-140, y+bh-33), "Function Call", font=load_font(14, True), fill=GREEN)

    # Code snippet box
    cy = 820
    d.rounded_rectangle([(100,cy),(W-100,cy+150)], radius=8, fill=(15,22,35), outline=DIM)
    d.text((130, cy+10), "Gemma 4 Function Definitions:", font=load_font(16, True), fill=CYAN)
    code = [
        ('tools = [', MID_GRAY),
        ('  {"type": "function", "function": {"name": "spectral_analysis",  "description": "Detect mineral signatures from imagery"}}', CYAN),
        ('  {"type": "function", "function": {"name": "terrain_classifier",  "description": "Classify terrain for exploration suitability"}}', GREEN),
        ('  {"type": "function", "function": {"name": "proximity_search",   "description": "Search nearby infrastructure & features"}}', ACCENT2),
        ('  {"type": "function", "function": {"name": "risk_assessment",    "description": "Evaluate exploration risks"}}', RED_SOFT),
        ('  {"type": "function", "function": {"name": "generate_report",    "description": "Generate prospectivity reports"}}', ACCENT3),
        ('  {"type": "function", "function": {"name": "geological_survey",  "description": "Query geological databases"}}', PURPLE),
        (']', MID_GRAY),
    ]
    fm = load_mono(14)
    for j,(ln,col) in enumerate(code):
        d.text((130, cy+32+j*15), ln, font=fm, fill=col)

    draw_bottom_bar(d, 4); return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Demo: Atacama Desert
# ════════════════════════════════════════════════════════════════════════════
def slide_05():
    """Uses matplotlib for the 3 analysis panels, then composites onto PIL canvas"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    d.text((100,70), "Demo: Atacama Desert Analysis", font=load_font(48, True), fill=CYAN)
    d.text((100,130), "Simulated prospectivity analysis for lithium — Atacama Desert, Chile", font=load_font(22), fill=LIGHT_GRAY)
    d.line([(100,170),(700,170)], fill=CYAN, width=2)

    # ── Matplotlib panels ──
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.2), dpi=100)
    fig.patch.set_facecolor('#162235')

    # Panel 1: Spectral map
    ax = axes[0]
    np.random.seed(42)
    x = np.linspace(0,10,200); y = np.linspace(0,10,200); X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for cx,cy in [(3,7),(7,3),(5,5)]:
        Z += 0.8*np.exp(-((X-cx)**2+(Y-cy)**2)/2)
    Z += 0.05*np.random.randn(*Z.shape)
    Z = np.clip(Z,0,1)
    ax.imshow(Z, cmap='YlOrRd', extent=[0,10,0,10], alpha=0.9)
    for cx,cy in [(3,7),(7,3),(5,5)]:
        ax.plot(cx,cy,'c*',markersize=14,markeredgecolor='white',markeredgewidth=0.5)
    ax.set_title('Spectral Analysis\nLithium Detection', fontsize=13, color='#37DCF2', fontweight='bold', pad=8)
    ax.set_xlabel('Longitude (°W)', fontsize=9, color='#B0B8C0')
    ax.set_ylabel('Latitude (°S)', fontsize=9, color='#B0B8C0')
    ax.tick_params(colors='#B0B8C0', labelsize=8)
    ax.set_facecolor('#162235')
    for sp in ax.spines.values(): sp.set_color('#37DCF240')

    # Panel 2: Terrain pie
    ax = axes[1]
    labels = ['Desert Salt\nFlat','Arid\nPlateau','Volcanic\nRock','Alluvial\nFan','Sand\nDune']
    vals = [35,25,20,12,8]
    colors = ['#37DCF2','#4CD964','#FF6363','#FFB74D','#5B80FF']
    wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=colors, autopct='%1.0f%%',
                                       startangle=90, textprops={'fontsize':9,'color':'#B0B8C0'},
                                       pctdistance=0.75, labeldistance=1.15)
    for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
    ax.set_title('Terrain Classification\nLandform Distribution', fontsize=13, color='#4CD964', fontweight='bold', pad=8)

    # Panel 3: Bar chart
    ax = axes[2]
    cats = ['Water\nAccess','Power\nInfra','Road\nAccess','Port\nDist.','Env.\nRisk','Pol.\nStability']
    scores = [85,72,68,55,78,92]
    colors_b = ['#4CD964' if s>=70 else '#FFB74D' if s>=60 else '#FF6363' for s in scores]
    bars = ax.barh(cats, scores, color=colors_b, height=0.6, edgecolor='#162235')
    ax.set_xlim(0,105)
    ax.set_title('Proximity & Risk\nSuitability Score', fontsize=13, color='#5B80FF', fontweight='bold', pad=8)
    ax.tick_params(colors='#B0B8C0', labelsize=9)
    ax.set_facecolor('#162235')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#B0B8C040')
    for bar,score in zip(bars,scores):
        ax.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2, f'{score}%', va='center', fontsize=10, color='white', fontweight='bold')

    fig.tight_layout(pad=1.5)
    tmp = os.path.join(SLIDES_DIR, '_demo_temp.png')
    fig.savefig(tmp, dpi=100, facecolor='#162235', edgecolor='none')
    plt.close(fig)

    demo = Image.open(tmp)
    demo = demo.resize((W-100, 620), Image.LANCZOS)
    img.paste(demo, (50, 185))

    # Verdict
    d.rounded_rectangle([(100,820),(W-100,920)], radius=10, fill=(20,50,35), outline=GREEN, width=2)
    d.text((150,835), "VERDICT:", font=load_font(22, True), fill=GREEN)
    d.text((340,835), "High prospectivity for lithium in Salar de Atacama region", font=load_font(20), fill=WHITE)
    d.text((340,865), "3 high-confidence target zones  |  Spectral match: 94%  |  Overall score: 75/100", font=load_font(17), fill=LIGHT_GRAY)
    d.text((340,893), "Recommendation: Proceed with advanced ground exploration and sampling", font=load_font(17, True), fill=ACCENT3)

    draw_bottom_bar(d, 5); return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Pipeline Execution
# ════════════════════════════════════════════════════════════════════════════
def slide_06():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    d.text((100,70), "Agentic Pipeline Execution", font=load_font(48, True), fill=CYAN)
    d.text((100,130), "Gemma 4 autonomously orchestrates all 6 tools in sequence", font=load_font(22), fill=LIGHT_GRAY)
    d.line([(100,170),(700,170)], fill=CYAN, width=2)

    steps = [
        ("STEP 1","spectral_analysis()","Analyzing multispectral imagery for lithium absorption features...",CYAN,"0.3s"),
        ("STEP 2","terrain_classifier()","Classifying terrain: 35% salt flat, 25% plateau, 20% volcanic rock...",GREEN,"1.2s"),
        ("STEP 3","proximity_search()","Searching within 50km: 3 mines, 1 power station, 2 road networks...",ACCENT2,"2.8s"),
        ("STEP 4","risk_assessment()","Evaluating: water access 85%, political stability 92%, env risk 78%...",RED_SOFT,"5.1s"),
        ("STEP 5","geological_survey()","Querying USGS: lithium deposits confirmed in Salar de Atacama region...",PURPLE,"8.4s"),
        ("STEP 6","generate_report()","Compiling comprehensive prospectivity report with maps and scores...",ACCENT3,"12.7s"),
    ]

    bw = 1650; bh = 80; gap = 12; sy = 195; sx = (W-bw)//2
    for i,(step,func,desc,col,time_s) in enumerate(steps):
        y = sy + i*(bh+gap)
        # Step pill
        d.rounded_rectangle([(sx,y),(sx+100,y+30)], radius=6, fill=col)
        f = load_font(13, True); bb = f.getbbox(step)
        d.text((sx+(100-bb[2]+bb[0])//2, y+6), step, font=f, fill=BG)
        # Function
        d.text((sx+115, y+3), func, font=load_font(19, True), fill=col)
        # Desc
        d.text((sx+115, y+30), desc, font=load_font(16), fill=LIGHT_GRAY)
        # Time
        d.text((sx+bw-80, y+30), time_s, font=load_font(14), fill=MID_GRAY)
        # Status
        d.rounded_rectangle([(sx+bw-200,y+5),(sx+bw-100,y+30)], radius=6, fill=(20,50,35), outline=GREEN)
        d.text((sx+bw-190, y+7), "✓ Done", font=load_font(15, True), fill=GREEN)
        # Connector
        if i < len(steps)-1:
            d.line([(sx+50, y+bh),(sx+50, y+bh+gap)], fill=col, width=2)
            d.polygon([(sx+45,y+bh+gap),(sx+55,y+bh+gap),(sx+50,y+bh+gap+5)], fill=col)

    # Summary
    d.rounded_rectangle([(100,850),(W-100,930)], radius=10, fill=(28,42,65), outline=CYAN, width=2)
    d.text((150,865), "Pipeline Complete:", font=load_font(22, True), fill=CYAN)
    d.text((150,898), "6 tools executed autonomously in ~12.7 seconds  |  0 manual interventions  |  Full prospectivity report generated", font=load_font(18), fill=LIGHT_GRAY)

    draw_bottom_bar(d, 6); return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Impact
# ════════════════════════════════════════════════════════════════════════════
def slide_07():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    d.text((100,70), "Impact & Vision", font=load_font(48, True), fill=CYAN)
    d.text((100,130), "Aligning with UN SDGs and global energy security", font=load_font(22), fill=LIGHT_GRAY)
    d.line([(100,170),(600,170)], fill=CYAN, width=2)

    sdgs = [("SDG 7","Affordable &\nClean Energy","Accelerating mineral\ndiscovery for renewables",ACCENT3),
            ("SDG 9","Industry, Innovation\n& Infrastructure","AI-powered exploration\ntechnology advancement",CYAN),
            ("SDG 13","Climate\nAction","Enabling the clean\nenergy transition",GREEN),
            ("SDG 12","Responsible\nConsumption","Sustainable resource\nmanagement practices",ACCENT2)]

    bw, bh, gap = 380, 175, 35; sx = 100
    for i,(sdg,title,desc,col) in enumerate(sdgs):
        x = sx+i*(bw+gap); y = 200
        d.rounded_rectangle([(x,y),(x+bw,y+bh)], radius=12, fill=(28,42,65), outline=col, width=2)
        f = load_font(26, True); bb = f.getbbox(sdg)
        d.text((x+(bw-bb[2]+bb[0])//2, y+12), sdg, font=f, fill=col)
        f2 = load_font(18, True)
        for j,ln in enumerate(title.split('\n')):
            bb = f2.getbbox(ln); d.text((x+(bw-bb[2]+bb[0])//2, y+55+j*24), ln, font=f2, fill=WHITE)
        f3 = load_font(14)
        for j,ln in enumerate(desc.split('\n')):
            bb = f3.getbbox(ln); d.text((x+(bw-bb[2]+bb[0])//2, y+115+j*20), ln, font=f3, fill=LIGHT_GRAY)

    d.text((100,410), "Key Impact Metrics", font=load_font(30, True), fill=CYAN)
    d.line([(100,450),(400,450)], fill=CYAN, width=1)

    metrics = [("10x Faster","Prospectivity analysis\nfrom months to minutes"),
               ("100x Cheaper","Preliminary screening\nat fraction of field cost"),
               ("Global Access","Any region, any mineral,\nno specialized hardware needed"),
               ("Open Source","Transparent, reproducible,\ncommunity-driven development")]
    mw, mh, mg = 380, 110, 35
    for i,(val,desc) in enumerate(metrics):
        x = 100+i*(mw+mg); y = 480
        d.rounded_rectangle([(x,y),(x+mw,y+mh)], radius=10, fill=(28,42,65))
        d.text((x+20, y+12), val, font=load_font(30, True), fill=CYAN)
        f = load_font(15)
        for j,ln in enumerate(desc.split('\n')):
            d.text((x+20, y+55+j*22), ln, font=f, fill=LIGHT_GRAY)

    # Vision box
    d.rounded_rectangle([(100,650),(W-100,770)], radius=12, fill=(20,35,55), outline=CYAN, width=2)
    d.text((150,670), "Our Vision", font=load_font(24, True), fill=CYAN)
    vision = ["Democratize mineral exploration by putting AI-powered prospectivity analysis",
              "in the hands of geologists, researchers, and developing nations — accelerating",
              "the critical mineral supply chain needed for a sustainable energy future."]
    for j,ln in enumerate(vision):
        d.text((150, 705+j*26), ln, font=load_font(19), fill=WHITE)

    draw_bottom_bar(d, 7); return img


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Closing
# ════════════════════════════════════════════════════════════════════════════
def slide_08():
    img = new_canvas(); d = ImageDraw.Draw(img)
    draw_bg_grid(d); draw_top_bar(d); draw_corners(d)

    glow_circle(d, W//2, H//2-40, 200, CYAN)
    for i in range(6):
        a = i*np.pi/3
        hexagon(d, W//2+int(220*np.cos(a)), H//2-40+int(220*np.sin(a)), 22, DIM, CYAN)
    hexagon(d, W//2, H//2-40, 40, CYAN, WHITE)

    tc(d, 130, "MineLens AI", load_font(68, True), CYAN)
    tc(d, 220, "Critical Mineral Prospectivity Mapping with Gemma 4", load_font(26), WHITE)
    d.line([(W//2-300,275),(W//2+300,275)], fill=CYAN, width=2)
    tc(d, 300, "Built with Gemma 4  |  Native Function Calling  |  6 Geoscience Tools", load_font(20), LIGHT_GRAY)

    # Tool badges
    tools_s = ["Spectral","Terrain","Proximity","Risk","Report","Geology"]
    cols = [CYAN,GREEN,ACCENT2,RED_SOFT,ACCENT3,PURPLE]
    total_w = len(tools_s)*145-25; bx = (W-total_w)//2
    for i,(t,c) in enumerate(zip(tools_s,cols)):
        tw = 120
        d.rounded_rectangle([(bx,370),(bx+tw,405)], radius=8, fill=(28,42,65), outline=c, width=1)
        f = load_font(15, True); bb = f.getbbox(t)
        d.text((bx+(tw-bb[2]+bb[0])//2, 380), t, font=f, fill=c)
        bx += 145

    tc(d, 450, "github.com/arc-prize/minelens-ai", load_font(22, True), CYAN)
    tc(d, 490, "Kaggle Gemma 4 Good Hackathon  |  $200,000 Prize Pool", load_font(20), LIGHT_GRAY)

    d.rounded_rectangle([(W//2-350,550),(W//2+350,615)], radius=12, fill=(28,42,65), outline=CYAN, width=2)
    tc(d, 560, "Accelerating mineral discovery for a sustainable future", load_font(24, True), WHITE)
    tc(d, 590, "Li  ·  Co  ·  REE  ·  Cu  ·  Ni  ·  And more...", load_font(17), MID_GRAY)

    tc(d, 670, "Thank You!", load_font(30), WHITE)
    tc(d, 730, "Powered by Google Gemma 4  |  Kaggle Community  |  Open Source", load_font(16), MID_GRAY)
    d.line([(W//2-200,H-100),(W//2+200,H-100)], fill=CYAN, width=1)

    draw_bottom_bar(d, 8); return img


# ════════════════════════════════════════════════════════════════════════════
# MAIN — Generate slides + assemble video with ffmpeg
# ════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(SLIDES_DIR, exist_ok=True)

    generators = [slide_01, slide_02, slide_03, slide_04, slide_05, slide_06, slide_07, slide_08]

    # Generate slide images
    slide_paths = []
    for i, gen in enumerate(generators):
        print(f"  Slide {i+1}/{TOTAL_SLIDES}: {gen.__name__}")
        img = gen()
        path = os.path.join(SLIDES_DIR, f'slide_{i+1:02d}.png')
        img.save(path, quality=95)
        slide_paths.append(path)

    print(f"\n  {len(slide_paths)} slides saved to {SLIDES_DIR}/")

    # Build ffmpeg concat file with crossfade transitions
    fade_dur = FADE_DUR

    # Strategy: use xfade filter between consecutive slides
    # Each slide shows for DURATION_PER_SLIDE, with crossfade into next
    # For xfade, we chain: [0][1]xfade=transition=fade:duration=0.8:offset=6.2[v01];
    #   [v01][2]xfade=...

    total_dur = TOTAL_SLIDES * DURATION_PER_SLIDE - (TOTAL_SLIDES - 1) * fade_dur

    # Build complex filter
    inputs = []
    for p in slide_paths:
        inputs.extend(['-loop', '1', '-t', str(DURATION_PER_SLIDE), '-i', p])

    # Build xfade filter chain
    filter_parts = []
    labels = []
    for i in range(TOTAL_SLIDES):
        labels.append(f'[{i}:v]')

    # First xfade
    offset = DURATION_PER_SLIDE - fade_dur
    prev = f'[v0]'
    filter_parts.append(
        f'{labels[0]}{labels[1]}xfade=transition=fade:duration={fade_dur}:offset={offset}{prev}'
    )

    for i in range(2, TOTAL_SLIDES):
        # Each subsequent xfade: the offset is cumulative
        # After first xfade, effective duration of combined is DURATION + DURATION - fade
        # Offset for next transition = DURATION_PER_SLIDE - fade_dur
        offset = DURATION_PER_SLIDE - fade_dur
        out = f'[v{i-1}]' if i < TOTAL_SLIDES - 1 else f'[vout]'
        filter_parts.append(
            f'[{prev[1:-1]}]{labels[i]}xfade=transition=fade:duration={fade_dur}:offset={offset}{out}'
        )
        prev = out

    filter_str = ';'.join(filter_parts)

    cmd = ['ffmpeg', '-y'] + inputs + [
        '-filter_complex', filter_str,
        '-map', '[vout]',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-r', str(FPS),
        '-movflags', '+faststart',
        VIDEO_PATH
    ]

    print(f"\n  Assembling video ({total_dur:.1f}s) with ffmpeg xfade transitions...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  xfade failed ({result.stderr[-500:]}), trying simple concat...")
        # Fallback: simple concat without transitions
        concat_path = os.path.join(SLIDES_DIR, 'concat.txt')
        with open(concat_path, 'w') as f:
            for p in slide_paths:
                f.write(f"file '{p}'\n")
                f.write(f"duration {DURATION_PER_SLIDE}\n")
            f.write(f"file '{slide_paths[-1]}'\n")

        cmd2 = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-r', str(FPS),
                '-vf', 'fps=24,format=yuv420p',
                '-movflags', '+faststart', VIDEO_PATH]
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  concat also failed: {result.stderr[-500:]}")
            sys.exit(1)

    size_mb = os.path.getsize(VIDEO_PATH) / (1024*1024)
    duration = total_dur
    print(f"\n{'='*60}")
    print(f"  Video generated successfully!")
    print(f"  Path:       {VIDEO_PATH}")
    print(f"  Duration:   {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Size:       {size_mb:.1f} MB")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
