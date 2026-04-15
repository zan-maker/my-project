#!/usr/bin/env python3
"""Generate 8 professional dashboard mockup frames for MineScope demo video."""

from PIL import Image, ImageDraw, ImageFont
import math
import os

# Constants
W, H = 1920, 1080
BG = "#0f172a"
BG_CARD = "#1e293b"
BG_CARD_LIGHT = "#334155"
TEAL = "#0d9488"
TEAL_LIGHT = "#14b8a6"
GOLD = "#d97706"
GOLD_LIGHT = "#f59e0b"
RED = "#ef4444"
GREEN = "#22c55e"
WHITE = "#f8fafc"
GRAY = "#94a3b8"
GRAY_DARK = "#475569"
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
LOGO_PATH = "/home/z/my-project/download/minescope-logo-480x480.png"
OUT_DIR = "/home/z/my-project/download/minescope-frames/"

# Load fonts
font_xl = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 64)
font_lg = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 48)
font_md = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 32)
font_md_r = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 32)
font_sm = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 24)
font_sm_b = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 24)
font_xs = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 18)
font_xs_b = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 18)
font_icon = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 40)
font_title = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 80)

# Load logo
logo_img = Image.open(LOGO_PATH).convert("RGBA")
logo_200 = logo_img.resize((200, 200), Image.LANCZOS)
logo_300 = logo_img.resize((300, 300), Image.LANCZOS)
logo_120 = logo_img.resize((120, 120), Image.LANCZOS)


def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_gradient_bg(img, color_top="#0f172a", color_bottom="#020617"):
    draw = ImageDraw.Draw(img)
    for y in range(H):
        r = int(int(color_top[1:3], 16) * (1 - y/H) + int(color_bottom[1:3], 16) * (y/H))
        g = int(int(color_top[3:5], 16) * (1 - y/H) + int(color_bottom[3:5], 16) * (y/H))
        b = int(int(color_top[5:7], 16) * (1 - y/H) + int(color_bottom[5:7], 16) * (y/H))
        draw.line([(0, y), (W, y)], fill=f"#{r:02x}{g:02x}{b:02x}")


def draw_tech_grid(img, opacity=20):
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, W, 80):
        draw.line([(x, 0), (x, H)], fill=(13, 148, 136, opacity), width=1)
    for y in range(0, H, 80):
        draw.line([(0, y), (W, y)], fill=(13, 148, 136, opacity), width=1)
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def draw_glow_circle(draw, cx, cy, r, color, alpha=60):
    for i in range(r, 0, -2):
        a = int(alpha * (i / r))
        draw.ellipse([cx-i, cy-i, cx+i, cy+i], fill=None, outline=color + f"{a:02x}" if len(color)==7 else color)


def paste_center(img, paste_img, y_offset=0):
    x = (W - paste_img.width) // 2
    y = y_offset
    if paste_img.mode == "RGBA":
        img.paste(paste_img, (x, y), paste_img)
    else:
        img.paste(paste_img, (x, y))


def draw_kpi_card(draw, x, y, w, h, mineral, price, change, risk, change_color):
    draw_rounded_rect(draw, (x, y, x+w, y+h), 12, BG_CARD)
    # Mineral name
    draw.text((x+20, y+15), mineral, fill=WHITE, font=font_sm_b)
    # Price
    draw.text((x+20, y+45), f"${price}/t", fill=WHITE, font=font_md)
    # Change
    draw.text((x+20, y+85), f"{change}%", fill=change_color, font=font_sm_b)
    # Risk badge
    draw_rounded_rect(draw, (x+w-90, y+15, x+w-10, y+45), 8, risk)
    draw.text((x+w-82, y+18), f"R:{risk_score}", fill=WHITE, font=font_xs_b)


risk_score = 72


def draw_line_chart(draw, x, y, w, h, datasets, labels=None):
    """Draw a multi-line chart."""
    draw_rounded_rect(draw, (x, y, x+w, y+h), 12, BG_CARD)
    draw.text((x+20, y+15), "12-Month Price Trends", fill=WHITE, font=font_sm_b)
    
    chart_x = x + 60
    chart_y = y + 50
    chart_w = w - 80
    chart_h = h - 90
    
    # Grid lines
    for i in range(5):
        gy = chart_y + i * chart_h // 4
        draw.line([(chart_x, gy), (chart_x + chart_w, gy)], fill=GRAY_DARK, width=1)
        val = 100 - i * 25
        draw.text((x + 10, gy - 10), f"{val}%", fill=GRAY, font=font_xs)
    
    # Month labels
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i, m in enumerate(months):
        mx = chart_x + i * chart_w // 11
        draw.text((mx - 10, chart_y + chart_h + 5), m, fill=GRAY, font=font_xs)
    
    # Draw lines
    colors = [TEAL_LIGHT, GOLD_LIGHT, "#a78bfa", "#fb7185", "#38bdf8"]
    for di, data in enumerate(datasets):
        points = []
        for i, val in enumerate(data):
            px = chart_x + i * chart_w // (len(data) - 1)
            py = chart_y + chart_h - (val / 100) * chart_h
            points.append((px, py))
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=colors[di % len(colors)], width=3)
        # End dot
        draw.ellipse([points[-1][0]-5, points[-1][1]-5, points[-1][0]+5, points[-1][1]+5], fill=colors[di % len(colors)])


def draw_sidebar(draw, x, y, w, h, active=0):
    draw.rectangle((x, y, x+w, y+h), fill="#0c1222")
    # Logo at top
    draw.text((x+15, y+20), "MineScope", fill=TEAL_LIGHT, font=font_sm_b)
    draw.line([(x+10, y+55), (x+w-10, y+55)], fill=GRAY_DARK, width=1)
    
    items = ["Dashboard", "Price Tracker", "Supply Chain", "Risk Analysis", "Reserves", "Companies", "ESG"]
    icons = ["◉", "◈", "◉", "◈", "◉", "◈", "◉"]
    for i, item in enumerate(items):
        iy = y + 75 + i * 45
        if i == active:
            draw_rounded_rect(draw, (x+5, iy-5, x+w-5, iy+30), 8, TEAL + "30")
            draw.rectangle((x+5, iy-5, x+8, iy+30), fill=TEAL)
            draw.text((x+18, iy), item, fill=TEAL_LIGHT, font=font_sm)
        else:
            draw.text((x+18, iy), item, fill=GRAY, font=font_sm)


# ============================================
# FRAME 1 - Title Card
# ============================================
def frame1():
    img = Image.new("RGB", (W, H), BG)
    draw_gradient_bg(img, "#0f172a", "#020617")
    draw_tech_grid(img, 15)
    
    # Centered glow
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for r in range(300, 0, -3):
        a = int(25 * (r / 300))
        od.ellipse([W//2-r, 380-r, W//2+r, 380+r], fill=(13, 148, 136, a))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    
    draw = ImageDraw.Draw(img)
    
    # Logo
    img.paste(logo_300, ((W - 300)//2, 140), logo_300)
    
    # Title
    title = "MineScope"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw)//2, 460), title, fill=WHITE, font=font_title)
    
    # Teal underline
    uw = tw
    draw.rectangle([(W - uw)//2, 555, (W + uw)//2, 558], fill=TEAL)
    
    # Subtitle line 1
    sub1 = "Critical Mineral Supply Chain Intelligence"
    bbox1 = draw.textbbox((0, 0), sub1, font=font_lg)
    sw1 = bbox1[2] - bbox1[0]
    draw.text(((W - sw1)//2, 580), sub1, fill=TEAL_LIGHT, font=font_lg)
    
    # Subtitle line 2
    sub2 = "Built with MeDo by Baidu"
    bbox2 = draw.textbbox((0, 0), sub2, font=font_md_r)
    sw2 = bbox2[2] - bbox2[0]
    draw.text(((W - sw2)//2, 645), sub2, fill=GOLD, font=font_md_r)
    
    # Bottom tagline
    tag = "Real-time commodity tracking  |  Geopolitical risk mapping  |  ESG monitoring"
    bbox3 = draw.textbbox((0, 0), tag, font=font_sm)
    sw3 = bbox3[2] - bbox3[0]
    draw.text(((W - sw3)//2, 740), tag, fill=GRAY, font=font_sm)
    
    # Version badge
    draw_rounded_rect(draw, (W//2-60, 800, W//2+60, 835), 20, TEAL)
    draw.text((W//2-35, 805), "v1.0", fill=WHITE, font=font_sm_b)
    
    img.save(os.path.join(OUT_DIR, "frame_01.png"))
    print("Frame 1 done")


# ============================================
# FRAME 2 - Problem Statement
# ============================================
def frame2():
    img = Image.new("RGB", (W, H), BG)
    draw_gradient_bg(img, "#0f172a", "#020617")
    draw_tech_grid(img, 10)
    draw = ImageDraw.Draw(img)
    
    # Section title
    draw.text((100, 60), "THE CHALLENGE", fill=TEAL_LIGHT, font=font_sm_b)
    draw.text((100, 95), "Why Critical Minerals Intelligence Matters", fill=WHITE, font=font_lg)
    
    cards = [
        {
            "icon": "◉",
            "title": "Supply Chain Opacity",
            "stat": "74%",
            "desc": "of critical mineral supply\nchains lack real-time visibility",
            "color": RED,
            "bar": 74
        },
        {
            "icon": "◈",
            "title": "Geopolitical Risk",
            "stat": "3 Countries",
            "desc": "control 80% of rare earth\nproduction globally",
            "color": GOLD,
            "bar": 80
        },
        {
            "icon": "◉",
            "title": "Data Fragmentation",
            "stat": "50+ Sources",
            "desc": "mining data scattered with\nno unified analytical view",
            "color": TEAL_LIGHT,
            "bar": 50
        }
    ]
    
    card_w = 500
    card_h = 550
    gap = 60
    start_x = (W - 3 * card_w - 2 * gap) // 2
    card_y = 200
    
    for i, card in enumerate(cards):
        cx = start_x + i * (card_w + gap)
        draw_rounded_rect(draw, (cx, card_y, cx + card_w, card_y + card_h), 16, BG_CARD)
        
        # Top accent line
        draw.rectangle([(cx+20, card_y+20), (cx+60, card_y+24)], fill=card["color"])
        
        # Icon circle
        draw.ellipse([cx+30, card_y+50, cx+90, card_y+110], fill=card["color"] + "20", outline=card["color"], width=2)
        draw.text((cx+45, card_y+60), card["icon"], fill=card["color"], font=font_icon)
        
        # Title
        draw.text((cx+110, card_y+55), card["title"], fill=WHITE, font=font_md)
        
        # Big stat
        draw.text((cx+30, card_y+140), card["stat"], fill=card["color"], font=font_xl)
        
        # Description
        lines = card["desc"].split("\n")
        for j, line in enumerate(lines):
            draw.text((cx+30, card_y+230 + j*30), line, fill=GRAY, font=font_sm)
        
        # Progress bar
        bar_y = card_y + 350
        draw_rounded_rect(draw, (cx+30, bar_y, cx+card_w-30, bar_y+16), 8, BG_CARD_LIGHT)
        draw_rounded_rect(draw, (cx+30, bar_y, cx+30 + int((card_w-60)*card["bar"]/100), bar_y+16), 8, card["color"])
        
        # Bar label
        draw.text((cx+30, bar_y+30), f"Risk Level: {card['bar']}%", fill=GRAY, font=font_xs)
    
    # Bottom quote
    draw.text((100, H-80), '"The energy transition will only succeed if we can see the full supply chain."', fill=GRAY, font=font_sm)
    
    img.save(os.path.join(OUT_DIR, "frame_02.png"))
    print("Frame 2 done")


# ============================================
# FRAME 3 - MeDo Conversation
# ============================================
def frame3():
    img = Image.new("RGB", (W, H), BG)
    draw_gradient_bg(img, "#0f172a", "#020617")
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.text((100, 30), "BUILDING WITH MEDO", fill=TEAL_LIGHT, font=font_sm_b)
    draw.text((100, 60), "From Conversation to Dashboard", fill=WHITE, font=font_lg)
    
    # Chat container
    chat_x, chat_y = 100, 140
    chat_w, chat_h = W - 200, 850
    draw_rounded_rect(draw, (chat_x, chat_y, chat_x + chat_w, chat_y + chat_h), 16, BG_CARD)
    
    # MeDo header bar
    draw_rounded_rect(draw, (chat_x, chat_y, chat_x + chat_w, chat_y + 60), 16, TEAL + "30")
    draw.rectangle((chat_x, chat_y + 44, chat_x + chat_w, chat_y + 60), fill=BG_CARD)
    img.paste(logo_120, (chat_x + 20, chat_y + 8), logo_120)
    draw.text((chat_x + 155, chat_y + 15), "MeDo — AI App Builder", fill=WHITE, font=font_sm_b)
    draw.text((chat_x + chat_w - 150, chat_y + 18), "● Active", fill=GREEN, font=font_sm)
    
    # User message 1
    msg_y = chat_y + 80
    draw_rounded_rect(draw, (chat_x + 500, msg_y, chat_x + chat_w - 30, msg_y + 120), 12, "#1e3a5f")
    draw.text((chat_x + 520, msg_y + 10), "👤 User", fill="#60a5fa", font=font_sm_b)
    draw.text((chat_x + 520, msg_y + 40), "Build a critical mineral supply chain dashboard with", fill=WHITE, font=font_sm)
    draw.text((chat_x + 520, msg_y + 65), "commodity prices, risk mapping, and ESG tracking.", fill=WHITE, font=font_sm)
    
    # MeDo response 1
    msg_y += 150
    draw_rounded_rect(draw, (chat_x + 30, msg_y, chat_x + chat_w - 500, msg_y + 160), 12, TEAL + "20")
    draw.text((chat_x + 50, msg_y + 10), "🤖 MeDo", fill=TEAL_LIGHT, font=font_sm_b)
    draw.text((chat_x + 50, msg_y + 40), "I'll create a comprehensive dashboard with", fill=WHITE, font=font_sm)
    draw.text((chat_x + 50, msg_y + 65), "7 modules: Dashboard Overview, Price Tracker,", fill=WHITE, font=font_sm)
    draw.text((chat_x + 50, msg_y + 90), "Supply Chain Map, Risk Analysis, Reserves,", fill=WHITE, font=font_sm)
    draw.text((chat_x + 50, msg_y + 115), "Company Comparison, and ESG Monitoring.", fill=WHITE, font=font_sm)
    
    # User message 2
    msg_y += 190
    draw_rounded_rect(draw, (chat_x + 500, msg_y, chat_x + chat_w - 30, msg_y + 80), 12, "#1e3a5f")
    draw.text((chat_x + 520, msg_y + 10), "👤 User", fill="#60a5fa", font=font_sm_b)
    draw.text((chat_x + 520, msg_y + 40), "Add real-time data for lithium, cobalt, nickel, rare earths, copper.", fill=WHITE, font=font_sm)
    
    # MeDo response 2
    msg_y += 110
    draw_rounded_rect(draw, (chat_x + 30, msg_y, chat_x + chat_w - 300, msg_y + 160), 12, TEAL + "20")
    draw.text((chat_x + 50, msg_y + 10), "🤖 MeDo", fill=TEAL_LIGHT, font=font_sm_b)
    draw.text((chat_x + 50, msg_y + 40), "Done! I've integrated price feeds, geopolitical risk", fill=WHITE, font=font_sm)
    draw.text((chat_x + 50, msg_y + 65), "scoring, ESG benchmarks, and interactive charts.", fill=WHITE, font=font_sm)
    draw.text((chat_x + 50, msg_y + 90), "Your dashboard is ready at minescope.app", fill=TEAL_LIGHT, font=font_sm_b)
    
    # "Generating..." indicator
    msg_y += 190
    draw.text((chat_x + 50, msg_y), "⏳ Generating full-stack application...", fill=GOLD, font=font_sm)
    # Progress bar
    bar_y = msg_y + 35
    draw_rounded_rect(draw, (chat_x + 50, bar_y, chat_x + 500, bar_y + 10), 5, BG_CARD_LIGHT)
    draw_rounded_rect(draw, (chat_x + 50, bar_y, chat_x + 420, bar_y + 10), 5, GREEN)
    draw.text((chat_x + 510, bar_y - 5), "84% Complete", fill=GREEN, font=font_xs)
    
    img.save(os.path.join(OUT_DIR, "frame_03.png"))
    print("Frame 3 done")


# ============================================
# FRAME 4 - Main Dashboard
# ============================================
def frame4():
    img = Image.new("RGB", (W, H), "#0c1222")
    draw = ImageDraw.Draw(img)
    
    # Sidebar
    draw_sidebar(draw, 0, 0, 220, H, active=0)
    
    main_x = 240
    
    # Top bar
    draw.rectangle((main_x, 0, W, 55), fill="#0f172a")
    draw.text((main_x + 20, 12), "Dashboard Overview", fill=WHITE, font=font_md)
    draw.text((W - 250, 15), "Last updated: 2 min ago", fill=GRAY, font=font_sm)
    
    # KPI Cards row
    minerals = [
        ("Lithium", "$14,850", "+2.4%", GREEN),
        ("Cobalt", "$33,420", "-1.2%", RED),
        ("Nickel", "$18,760", "+0.8%", GREEN),
        ("Rare Earths", "$62,300", "+5.1%", GREEN),
        ("Copper", "$9,240", "+1.7%", GREEN),
    ]
    
    kpi_w = (W - main_x - 40) // 5 - 12
    for i, (name, price, change, color) in enumerate(minerals):
        kx = main_x + 15 + i * (kpi_w + 12)
        ky = 70
        draw_rounded_rect(draw, (kx, ky, kx + kpi_w, ky + 110), 10, BG_CARD)
        draw.text((kx + 12, ky + 8), name, fill=GRAY, font=font_xs_b)
        draw.text((kx + 12, ky + 32), price, fill=WHITE, font=font_md)
        draw.text((kx + 12, ky + 72), change, fill=color, font=font_sm_b)
        # Mini sparkline
        spark_y = ky + 85
        pts = [(kx+12, spark_y+15), (kx+30, spark_y+10), (kx+50, spark_y+18), (kx+70, spark_y+5), (kx+90, spark_y+12)]
        for j in range(len(pts)-1):
            draw.line([pts[j], pts[j+1]], fill=color, width=2)
    
    # Left chart area
    chart_data = [
        [45, 48, 52, 50, 55, 58, 56, 60, 62, 65, 68, 72],  # Lithium
        [70, 68, 72, 75, 73, 70, 68, 65, 63, 60, 58, 55],  # Cobalt
        [55, 56, 54, 57, 58, 60, 62, 61, 63, 65, 64, 66],  # Nickel
        [40, 42, 45, 50, 55, 58, 62, 65, 68, 72, 75, 78],  # REE
        [30, 32, 35, 38, 40, 42, 44, 45, 47, 48, 50, 52],  # Copper
    ]
    
    draw_line_chart(draw, main_x + 15, 195, 950, 380, chart_data)
    
    # Right panel - Supply Chain Spotlight
    sp_x = main_x + 985
    sp_y = 195
    sp_w = W - sp_x - 15
    draw_rounded_rect(draw, (sp_x, sp_y, sp_x + sp_w, sp_y + 380), 12, BG_CARD)
    draw.text((sp_x + 15, sp_y + 15), "Supply Chain Spotlight", fill=WHITE, font=font_sm_b)
    
    spotlights = [
        ("Lithium Carbonate", "Chile → China → Battery Mfg", TEAL_LIGHT),
        ("Cobalt Sulfate", "DRC → China → Korea → EVs", RED),
        ("Rare Earth Oxides", "Myanmar → China → Tech Mfg", GOLD),
        ("Nickel Matte", "Indo → China → Stainless Steel", TEAL_LIGHT),
        ("Copper Cathode", "Chile → Global → Infrastructure", GREEN),
    ]
    
    for i, (mineral, route, color) in enumerate(spotlights):
        sy = sp_y + 55 + i * 60
        draw.rounded_rectangle((sp_x+10, sy, sp_x+sp_w-10, sy+50), radius=8, fill=BG_CARD_LIGHT)
        draw.text((sp_x+20, sy+5), mineral, fill=WHITE, font=font_xs_b)
        draw.text((sp_x+20, sy+28), route, fill=GRAY, font=font_xs)
        draw.ellipse([sp_x+sp_w-30, sy+15, sp_x+sp_w-14, sy+31], fill=color)
    
    # Bottom alerts ticker
    ticker_y = 590
    draw_rounded_rect(draw, (main_x + 15, ticker_y, W - 15, ticker_y + 80), 12, BG_CARD)
    draw.text((main_x + 30, ticker_y + 10), "🔴 ALERTS", fill=RED, font=font_sm_b)
    alerts = [
        "China imposes new REE export quotas — 15% reduction",
        "DRC mining strike affects 30% of cobalt output",
        "Chile lithium production up 8% QoQ",
        "Australia nickel mine expansion approved",
        "EU Critical Minerals Act takes effect January 2025",
    ]
    alert_text = "  │  ".join(alerts)
    draw.text((main_x + 30, ticker_y + 42), alert_text, fill=GRAY, font=font_xs)
    
    # Bottom panels row
    bp_y = 690
    bp_w = (W - main_x - 40) // 2 - 6
    
    # Top Risks panel
    draw_rounded_rect(draw, (main_x + 15, bp_y, main_x + 15 + bp_w, bp_y + 370), 12, BG_CARD)
    draw.text((main_x + 30, bp_y + 15), "Top Risk Countries", fill=WHITE, font=font_sm_b)
    
    countries = [
        ("China", "REE Processing", "94", RED),
        ("DRC", "Cobalt Mining", "89", RED),
        ("Myanmar", "REE Mining", "82", GOLD),
        ("Russia", "Palladium", "78", GOLD),
        ("Indonesia", "Nickel Ore", "65", GOLD),
        ("Chile", "Lithium", "38", GREEN),
        ("Australia", "Lithium", "32", GREEN),
        ("Canada", "Nickel", "25", GREEN),
    ]
    
    for i, (country, mineral, score, color) in enumerate(countries):
        cy = bp_y + 50 + i * 38
        draw.text((main_x + 30, cy), country, fill=WHITE, font=font_xs_b)
        draw.text((main_x + 130, cy), mineral, fill=GRAY, font=font_xs)
        # Risk bar
        bar_x = main_x + 280
        bar_w = bp_w - 340
        draw_rounded_rect(draw, (bar_x, cy+5, bar_x + bar_w, cy+18), 6, BG_CARD_LIGHT)
        draw_rounded_rect(draw, (bar_x, cy+5, bar_x + int(bar_w*int(score)/100), cy+18), 6, color)
        draw.text((bar_x + bar_w + 10, cy), score, fill=color, font=font_xs_b)
    
    # ESG Quick View panel
    draw_rounded_rect(draw, (main_x + 15 + bp_w + 12, bp_y, main_x + 15 + 2*bp_w + 12, bp_y + 370), 12, BG_CARD)
    draw.text((main_x + 30 + bp_w + 12, bp_y + 15), "ESG Quick View", fill=WHITE, font=font_sm_b)
    
    esg_items = [
        ("Environmental", 78, TEAL_LIGHT),
        ("Social", 65, GOLD),
        ("Governance", 82, GREEN),
    ]
    for i, (label, score, color) in enumerate(esg_items):
        ey = bp_y + 50 + i * 80
        draw.text((main_x + 30 + bp_w + 12, ey), label, fill=WHITE, font=font_sm_b)
        # Circular score indicator
        cx_c = main_x + 30 + bp_w + 12 + bp_w - 80
        cy_c = ey + 20
        draw.ellipse([cx_c-30, cy_c-30, cx_c+30, cy_c+30], fill=None, outline=color, width=4)
        draw.text((cx_c-20, cy_c-12), str(score), fill=WHITE, font=font_md_b if score > 99 else font_md)
        # Bar
        bar_x2 = main_x + 30 + bp_w + 12
        bar_w2 = bp_w - 160
        draw_rounded_rect(draw, (bar_x2, ey+35, bar_x2 + bar_w2, ey+50), 8, BG_CARD_LIGHT)
        draw_rounded_rect(draw, (bar_x2, ey+35, bar_x2 + int(bar_w2*score/100), ey+50), 8, color)
    
    # Top companies
    draw.text((main_x + 30 + bp_w + 12, bp_y + 300), "Top Companies", fill=WHITE, font=font_sm_b)
    companies = ["Albemarle", "SQM", "Glencore", "BHP"]
    for i, comp in enumerate(companies):
        cx = main_x + 30 + bp_w + 12 + i * 150
        draw.text((cx, bp_y + 335), f"● {comp}", fill=TEAL_LIGHT if i < 2 else GRAY, font=font_xs)
    
    img.save(os.path.join(OUT_DIR, "frame_04.png"))
    print("Frame 4 done")


# ============================================
# FRAME 5 - Price Tracker
# ============================================
def frame5():
    img = Image.new("RGB", (W, H), "#0c1222")
    draw = ImageDraw.Draw(img)
    
    # Sidebar
    draw_sidebar(draw, 0, 0, 220, H, active=1)
    
    main_x = 240
    
    # Top bar
    draw.rectangle((main_x, 0, W, 55), fill="#0f172a")
    draw.text((main_x + 20, 12), "Price Tracker", fill=WHITE, font=font_md)
    
    # Time range selector
    ranges = ["1M", "3M", "6M", "1Y", "ALL"]
    rx = main_x + 500
    for i, r in enumerate(ranges):
        fill = TEAL if r == "1Y" else BG_CARD
        draw_rounded_rect(draw, (rx + i*65, 10, rx + i*65 + 55, 42), 8, fill)
        draw.text((rx + i*65 + 15, 14), r, fill=WHITE, font=font_xs_b)
    
    # Mineral filter pills
    minerals_filter = ["All", "Lithium", "Cobalt", "Nickel", "Rare Earths", "Copper"]
    px = main_x + 15
    for i, m in enumerate(minerals_filter):
        fill = TEAL if i == 0 else BG_CARD
        draw_rounded_rect(draw, (px, 65, px + (120 if i > 0 else 60), 95), 8, fill)
        draw.text((px + 10, 70), m, fill=WHITE, font=font_xs_b)
        px += 130 if i > 0 else 70
    
    # Large chart
    chart_data = [
        [45, 48, 52, 50, 55, 58, 56, 60, 62, 65, 68, 72],
        [70, 68, 72, 75, 73, 70, 68, 65, 63, 60, 58, 55],
        [55, 56, 54, 57, 58, 60, 62, 61, 63, 65, 64, 66],
        [40, 42, 45, 50, 55, 58, 62, 65, 68, 72, 75, 78],
        [30, 32, 35, 38, 40, 42, 44, 45, 47, 48, 50, 52],
    ]
    draw_line_chart(draw, main_x + 15, 110, W - main_x - 30, 480, chart_data)
    
    # Legend
    legend_items = [("Lithium", TEAL_LIGHT), ("Cobalt", "#a78bfa"), ("Nickel", GOLD_LIGHT), ("Rare Earths", "#fb7185"), ("Copper", "#38bdf8")]
    lx = main_x + 50
    for i, (name, color) in enumerate(legend_items):
        draw.rectangle((lx + i*160, 610, lx + i*160 + 20, 625), fill=color)
        draw.text((lx + i*160 + 28, 607), name, fill=WHITE, font=font_xs)
    
    # Price summary table
    table_y = 640
    draw_rounded_rect(draw, (main_x + 15, table_y, W - 15, table_y + 400), 12, BG_CARD)
    draw.text((main_x + 30, table_y + 15), "Price Summary — Last 12 Months", fill=WHITE, font=font_sm_b)
    
    # Table header
    headers = ["Mineral", "Current", "12M Ago", "Change", "Low", "High", "Avg", "Trend"]
    col_widths = [180, 150, 150, 120, 130, 130, 130, 130]
    hx = main_x + 30
    for i, (h, cw) in enumerate(zip(headers, col_widths)):
        draw.text((hx, table_y + 50), h, fill=GRAY, font=font_xs_b)
        hx += cw
    
    draw.line([(main_x + 30, table_y + 75), (W - 30, table_y + 75)], fill=GRAY_DARK, width=1)
    
    table_data = [
        ("Lithium Carbonate", "$14,850/t", "$14,500/t", "+2.4%", "$12,100", "$16,200", "$14,300", "↑"),
        ("Cobalt Sulfate", "$33,420/t", "$33,830/t", "-1.2%", "$28,500", "$36,100", "$32,800", "↓"),
        ("Nickel Matte", "$18,760/t", "$18,610/t", "+0.8%", "$16,200", "$20,100", "$18,400", "↑"),
        ("Rare Earth Oxides", "$62,300/t", "$59,280/t", "+5.1%", "$48,000", "$65,700", "$57,200", "↑↑"),
        ("Copper Cathode", "$9,240/t", "$9,085/t", "+1.7%", "$7,800", "$10,200", "$8,900", "↑"),
    ]
    
    change_colors = [GREEN, RED, GREEN, GREEN, GREEN]
    
    for row_i, row in enumerate(table_data):
        ry = table_y + 85 + row_i * 50
        hx = main_x + 30
        for col_i, (val, cw) in enumerate(zip(row, col_widths)):
            color = WHITE
            if col_i == 3:
                color = change_colors[row_i]
            elif col_i == 7:
                color = change_colors[row_i]
            draw.text((hx, ry + 5), val, fill=color, font=font_xs_b if col_i == 0 else font_xs)
            hx += cw
        if row_i < len(table_data) - 1:
            draw.line([(main_x + 30, ry + 45), (W - 30, ry + 45)], fill=GRAY_DARK, width=1)
    
    img.save(os.path.join(OUT_DIR, "frame_05.png"))
    print("Frame 5 done")


# ============================================
# FRAME 6 - Risk Analysis & Supply Chain Map
# ============================================
def frame6():
    img = Image.new("RGB", (W, H), "#0c1222")
    draw = ImageDraw.Draw(img)
    
    # Sidebar
    draw_sidebar(draw, 0, 0, 220, H, active=3)
    
    main_x = 240
    
    # Top bar
    draw.rectangle((main_x, 0, W, 55), fill="#0f172a")
    draw.text((main_x + 20, 12), "Risk Analysis & Supply Chain", fill=WHITE, font=font_md)
    
    # Left: Radar chart (drawn as polygon)
    radar_x = main_x + 15
    radar_y = 75
    radar_w = 580
    radar_h = 500
    draw_rounded_rect(draw, (radar_x, radar_y, radar_x + radar_w, radar_y + radar_h), 12, BG_CARD)
    draw.text((radar_x + 20, radar_y + 15), "Risk Radar — Rare Earth Elements", fill=WHITE, font=font_sm_b)
    
    # Draw radar
    cx_r = radar_x + radar_w // 2
    cy_r = radar_y + radar_h // 2 + 20
    radius = 180
    
    axes = ["Geopolitical", "Environmental", "Regulatory", "Infrastructure", "Labor", "Concentration"]
    values = [0.94, 0.72, 0.85, 0.68, 0.78, 0.92]  # Normalized risk scores
    n = len(axes)
    
    # Draw grid rings
    for ring in range(1, 5):
        r = radius * ring // 4
        points = []
        for i in range(n):
            angle = math.pi / 2 + 2 * math.pi * i / n
            px = cx_r + int(r * math.cos(angle))
            py = cy_r - int(r * math.sin(angle))
            points.append((px, py))
        for i in range(len(points)):
            draw.line([points[i], points[(i+1) % n]], fill=GRAY_DARK, width=1)
    
    # Draw axis lines
    for i in range(n):
        angle = math.pi / 2 + 2 * math.pi * i / n
        ex = cx_r + int(radius * math.cos(angle))
        ey = cy_r - int(radius * math.sin(angle))
        draw.line([(cx_r, cy_r), (ex, ey)], fill=GRAY_DARK, width=1)
        # Label
        lx = cx_r + int((radius + 30) * math.cos(angle))
        ly = cy_r - int((radius + 30) * math.sin(angle))
        draw.text((lx - 50, ly - 10), axes[i], fill=GRAY, font=font_xs)
    
    # Draw data polygon
    data_points = []
    for i in range(n):
        angle = math.pi / 2 + 2 * math.pi * i / n
        r = radius * values[i]
        px = cx_r + int(r * math.cos(angle))
        py = cy_r - int(r * math.sin(angle))
        data_points.append((px, py))
    
    # Fill polygon
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.polygon(data_points, fill=(239, 68, 68, 40), outline=(239, 68, 68, 200))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    draw = ImageDraw.Draw(img)
    
    for i in range(len(data_points)):
        draw.ellipse([data_points[i][0]-5, data_points[i][1]-5, data_points[i][0]+5, data_points[i][1]+5], fill=RED)
    
    # Overall risk score in center
    draw.ellipse([cx_r-35, cy_r-35, cx_r+35, cy_r+35], fill=RED)
    draw.text((cx_r-18, cy_r-15), "82", fill=WHITE, font=font_md)
    draw.text((cx_r-12, cy_r+18), "HIGH", fill=WHITE, font=font_xs_b)
    
    # Right: Country Risk Grid
    grid_x = main_x + 615
    grid_y = 75
    grid_w = W - grid_x - 15
    grid_h = 500
    draw_rounded_rect(draw, (grid_x, grid_y, grid_x + grid_w, grid_y + grid_h), 12, BG_CARD)
    draw.text((grid_x + 20, grid_y + 15), "Country Risk Scores", fill=WHITE, font=font_sm_b)
    
    country_risks = [
        ("China", "REE Processing", 94, RED),
        ("DRC", "Cobalt Mining", 89, RED),
        ("Myanmar", "REE Mining", 82, GOLD),
        ("Russia", "Palladium/Nickel", 78, GOLD),
        ("Indonesia", "Nickel Smelting", 65, GOLD),
        ("Chile", "Lithium/Copper", 38, GREEN),
        ("Australia", "Lithium/REE", 32, GREEN),
        ("Canada", "Nickel/Cobalt", 25, GREEN),
    ]
    
    # Table header
    ch_x = grid_x + 20
    headers_c = ["Country", "Focus", "Risk"]
    draw.text((ch_x, grid_y + 50), "Country", fill=GRAY, font=font_xs_b)
    draw.text((ch_x + 140, grid_y + 50), "Focus", fill=GRAY, font=font_xs_b)
    draw.text((ch_x + 380, grid_y + 50), "Risk Score", fill=GRAY, font=font_xs_b)
    draw.line([(ch_x, grid_y + 72), (grid_x + grid_w - 20, grid_y + 72)], fill=GRAY_DARK, width=1)
    
    for i, (country, focus, score, color) in enumerate(country_risks):
        ry = grid_y + 80 + i * 48
        draw.text((ch_x, ry + 5), country, fill=WHITE, font=font_sm_b)
        draw.text((ch_x + 140, ry + 5), focus, fill=GRAY, font=font_sm)
        # Risk bar
        bar_bx = ch_x + 380
        bar_bw = grid_w - 450
        draw_rounded_rect(draw, (bar_bx, ry+10, bar_bx + bar_bw, ry+25), 6, BG_CARD_LIGHT)
        draw_rounded_rect(draw, (bar_bx, ry+10, bar_bx + int(bar_bw*score/100), ry+25), 6, color)
        draw.text((bar_bx + bar_bw + 10, ry + 3), str(score), fill=color, font=font_sm_b)
        # Flag emoji
        flag = "🔴" if score >= 75 else ("🟡" if score >= 50 else "🟢")
        draw.text((ch_x + 310, ry + 5), flag, fill=WHITE, font=font_sm)
    
    # Alert callout box
    alert_y = grid_y + grid_h + 20
    draw_rounded_rect(draw, (main_x + 15, alert_y, W - 15, alert_y + 130), 12, "#3b1515")
    draw.text((main_x + 35, alert_y + 15), "⚠  CRITICAL ALERT", fill=RED, font=font_md)
    draw.text((main_x + 35, alert_y + 55), "China controls 60% of rare earth processing capacity — a critical vulnerability", fill=WHITE, font=font_sm)
    draw.text((main_x + 35, alert_y + 82), "for Western technology and defense supply chains.", fill=WHITE, font=font_sm)
    
    # Additional context boxes
    ctx_y = alert_y + 150
    ctx_w = (W - main_x - 40) // 2 - 6
    contexts = [
        ("Supply Concentration", "Top 3 producers control 80%\nof global rare earth supply", RED),
        ("Processing Dependency", "90% of REE processing occurs\nin China, creating single point of failure", GOLD),
    ]
    for i, (title, desc, color) in enumerate(contexts):
        bx = main_x + 15 + i * (ctx_w + 12)
        draw_rounded_rect(draw, (bx, ctx_y, bx + ctx_w, ctx_y + 160), 12, BG_CARD)
        draw.text((bx + 15, ctx_y + 15), title, fill=color, font=font_sm_b)
        for j, line in enumerate(desc.split("\n")):
            draw.text((bx + 15, ctx_y + 50 + j * 28), line, fill=GRAY, font=font_xs)
    
    img.save(os.path.join(OUT_DIR, "frame_06.png"))
    print("Frame 6 done")


# ============================================
# FRAME 7 - Company Comparison & ESG
# ============================================
def frame7():
    img = Image.new("RGB", (W, H), "#0c1222")
    draw = ImageDraw.Draw(img)
    
    # Sidebar
    draw_sidebar(draw, 0, 0, 220, H, active=5)
    
    main_x = 240
    
    # Top bar
    draw.rectangle((main_x, 0, W, 55), fill="#0f172a")
    draw.text((main_x + 20, 12), "Company Comparison & ESG", fill=WHITE, font=font_md)
    
    # Company comparison table
    draw_rounded_rect(draw, (main_x + 15, 70, W - 15, 530), 12, BG_CARD)
    draw.text((main_x + 30, 85), "Company Benchmarking", fill=WHITE, font=font_sm_b)
    
    # Headers
    comp_headers = ["Company", "HQ", "Production", "Reserves", "ESG Score", "Market Cap", "Trend"]
    comp_col_w = [200, 100, 180, 180, 150, 180, 150]
    hx = main_x + 30
    for h, cw in zip(comp_headers, comp_col_w):
        draw.text((hx, 120), h, fill=GRAY, font=font_xs_b)
        hx += cw
    draw.line([(main_x + 30, 145), (W - 30, 145)], fill=GRAY_DARK, width=1)
    
    companies = [
        ("Albemarle", "USA", "95K t Li", "12.4M t", 87, "$14.2B", "↑", GREEN),
        ("SQM", "Chile", "180K t Li", "8.7M t", 79, "$8.5B", "↑", GREEN),
        ("Glencore", "Switz.", "42K t Co", "3.1M t", 71, "$52.1B", "→", GRAY),
        ("BHP", "Australia", "1.7M t Ni", "6.2M t", 83, "$148B", "↑", GREEN),
        ("Lynas", "Australia", "12K t REE", "4.8M t", 76, "$5.2B", "↑↑", GREEN),
        ("CMOC", "China", "65K t Co", "8.9M t", 58, "$22.8B", "↓", RED),
    ]
    
    for i, (comp, hq, prod, reserves, esg, mcap, trend, tcolor) in enumerate(companies):
        ry = 155 + i * 55
        hx = main_x + 30
        vals = [comp, hq, prod, reserves, str(esg), mcap, trend]
        for j, (v, cw) in enumerate(zip(vals, comp_col_w)):
            color = WHITE
            if j == 4:
                color = GREEN if esg >= 75 else (GOLD if esg >= 60 else RED)
            elif j == 6:
                color = tcolor
            font = font_sm_b if j == 0 else font_xs
            draw.text((hx, ry + 8), v, fill=color, font=font_sm_b if j in [0, 4, 6] else font_xs)
            hx += cw
        if i < len(companies) - 1:
            draw.line([(main_x + 30, ry + 48), (W - 30, ry + 48)], fill=GRAY_DARK, width=1)
    
    # ESG Pillar Cards
    esg_y = 550
    esg_w = (W - main_x - 55) // 3
    
    pillars = [
        ("Environmental", 78, TEAL_LIGHT, [
            ("Carbon Emissions", "↓ 12%", GREEN),
            ("Water Usage", "→ Stable", GRAY),
            ("Land Rehabilitation", "↑ 8%", GREEN),
            ("Biodiversity", "↓ 3%", RED),
        ]),
        ("Social", 65, GOLD, [
            ("Worker Safety", "↑ 15%", GREEN),
            ("Community Impact", "→ Stable", GRAY),
            ("Local Employment", "↑ 22%", GREEN),
            ("Human Rights", "↓ 5%", RED),
        ]),
        ("Governance", 82, GREEN, [
            ("Board Diversity", "↑ 10%", GREEN),
            ("Transparency", "↑ 18%", GREEN),
            ("Anti-corruption", "→ Stable", GRAY),
            ("Tax Compliance", "↑ 5%", GREEN),
        ]),
    ]
    
    for i, (title, score, color, metrics) in enumerate(pillars):
        px = main_x + 15 + i * (esg_w + 12)
        draw_rounded_rect(draw, (px, esg_y, px + esg_w, esg_y + 470), 12, BG_CARD)
        
        # Title and score
        draw.text((px + 20, esg_y + 15), title, fill=WHITE, font=font_md)
        
        # Big score circle
        cx_e = px + esg_w - 70
        cy_e = esg_y + 45
        draw.ellipse([cx_e-35, cy_e-35, cx_e+35, cy_e+35], fill=None, outline=color, width=5)
        draw.text((cx_e - 18, cy_e - 15), str(score), fill=color, font=font_md)
        
        # Score bar
        bar_bx = px + 20
        bar_bw = esg_w - 180
        draw_rounded_rect(draw, (bar_bx, esg_y + 60, bar_bx + bar_bw, esg_y + 78), 8, BG_CARD_LIGHT)
        draw_rounded_rect(draw, (bar_bx, esg_y + 60, bar_bx + int(bar_bw*score/100), esg_y + 78), 8, color)
        
        draw.line([(px + 15, esg_y + 95), (px + esg_w - 15, esg_y + 95)], fill=GRAY_DARK, width=1)
        
        # Metrics
        for j, (metric, trend, tcolor) in enumerate(metrics):
            my = esg_y + 110 + j * 80
            draw.text((px + 20, my), metric, fill=WHITE, font=font_sm_b)
            draw.text((px + esg_w - 100, my), trend, fill=tcolor, font=font_sm_b)
            # Mini progress
            draw_rounded_rect(draw, (px + 20, my + 30, px + esg_w - 20, my + 42), 6, BG_CARD_LIGHT)
            fill_pct = 0.5 + (j * 0.12)
            if "↓" in trend and "Emissions" not in metric:
                fill_pct = 0.7 - j * 0.05
            draw_rounded_rect(draw, (px + 20, my + 30, px + 20 + int((esg_w-40)*fill_pct), my + 42), 6, tcolor)
    
    img.save(os.path.join(OUT_DIR, "frame_07.png"))
    print("Frame 7 done")


# ============================================
# FRAME 8 - Closing Card
# ============================================
def frame8():
    img = Image.new("RGB", (W, H), BG)
    draw_gradient_bg(img, "#0f172a", "#020617")
    draw_tech_grid(img, 12)
    
    # Glow effect
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for r in range(400, 0, -3):
        a = int(20 * (r / 400))
        od.ellipse([W//2-r, H//2-100-r, W//2+r, H//2-100+r], fill=(13, 148, 136, a))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    
    draw = ImageDraw.Draw(img)
    
    # Logo
    img.paste(logo_300, ((W - 300)//2, 120), logo_300)
    
    # Main tagline
    tag1 = "Built entirely with MeDo"
    bbox = draw.textbbox((0, 0), tag1, font=font_xl)
    tw1 = bbox[2] - bbox[0]
    draw.text(((W - tw1)//2, 450), tag1, fill=WHITE, font=font_xl)
    
    tag2 = "From Conversation to Production in Minutes"
    bbox2 = draw.textbbox((0, 0), tag2, font=font_md_r)
    tw2 = bbox2[2] - bbox2[0]
    draw.text(((W - tw2)//2, 535), tag2, fill=TEAL_LIGHT, font=font_md_r)
    
    # Divider
    draw.rectangle([(W//2 - 100, 590), (W//2 + 100, 593)], fill=TEAL)
    
    # GitHub link
    gh = "github.com/zan-maker/minescope"
    bbox3 = draw.textbbox((0, 0), gh, font=font_md_r)
    tw3 = bbox3[2] - bbox3[0]
    draw_rounded_rect(draw, ((W - tw3)//2 - 30, 620, (W + tw3)//2 + 30, 670), 12, BG_CARD)
    draw.text(((W - tw3)//2, 630), gh, fill=GOLD, font=font_md_r)
    
    # Hashtag
    hashtag = "#BuiltWithMeDo"
    bbox4 = draw.textbbox((0, 0), hashtag, font=font_lg)
    tw4 = bbox4[2] - bbox4[0]
    draw.text(((W - tw4)//2, 700), hashtag, fill=GOLD_LIGHT, font=font_lg)
    
    # Bottom tags
    tags = ["Critical Minerals", "Supply Chain", "AI Dashboard", "Baidu MeDo", "ESG", "Geopolitical Risk"]
    tag_y = 810
    total_w = sum(draw.textbbox((0, 0), f"  {t}  ", font=font_xs_b)[2] for t in tags) + 20 * (len(tags) - 1)
    tx = (W - total_w) // 2
    for t in tags:
        bbox_t = draw.textbbox((0, 0), f"  {t}  ", font=font_xs_b)
        tw_t = bbox_t[2] - bbox_t[0]
        draw_rounded_rect(draw, (tx, tag_y, tx + tw_t, tag_y + 30), 15, TEAL + "30")
        draw.text((tx, tag_y + 5), f"  {t}  ", fill=TEAL_LIGHT, font=font_xs_b)
        tx += tw_t + 20
    
    # Copyright
    copy_text = "© 2025 MineScope | Powered by MeDo AI"
    bbox5 = draw.textbbox((0, 0), copy_text, font=font_xs)
    tw5 = bbox5[2] - bbox5[0]
    draw.text(((W - tw5)//2, 870), copy_text, fill=GRAY_DARK, font=font_xs)
    
    img.save(os.path.join(OUT_DIR, "frame_08.png"))
    print("Frame 8 done")


# Generate all frames
print("Generating frames...")
frame1()
frame2()
frame3()
frame4()
frame5()
frame6()
frame7()
frame8()
print("All 8 frames generated!")
