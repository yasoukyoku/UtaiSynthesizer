# -*- coding: utf-8 -*-
"""Muno 品牌资产生成器(方案A「声波音符」)。

产出:
  src-tauri/icons/icon.png            512×512  (RGBA)
  src-tauri/icons/64x64.png           64×64
  src-tauri/icons/32x32.png           32×32
  src-tauri/icons/128x128.png         128×128
  src-tauri/icons/128x128@2x.png      256×256
  src-tauri/icons/icon.ico            多尺寸 16/24/32/48/64/128/256
  src-tauri/icons/Square*.png/StoreLogo.png  (按现有尺寸重绘)
  src-tauri/installer/header.bmp      150×57  (NSIS 顶部横条)
  src-tauri/installer/sidebar.bmp     164×314 (NSIS 左侧竖幅)
  public/icon.png                     512×512 (favicon)

设计:圆角深空蓝方底 #1E2A4A,白色八分音符(符尾化为向上折线=声波),
右下角荧光青圆点 #00D4AA。4× 超采样绘制后 LANCZOS 缩放。
"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_font(size, bold=True, cjk=False):
    """Windows 字体,带回退。"""
    cands = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    if cjk:
        cands = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"] + cands
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()

DEEP = (30, 42, 74, 255)        # #1E2A4A 深空蓝
CYAN = (0, 212, 170, 255)       # #00D4AA 荧光青
WHITE = (255, 255, 255, 255)

# 逻辑坐标系:1024×1024,内部按 4× 渲染。
S = 1024
SS = 4
CANVAS = S * SS


def rrect(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def thick_polyline(draw, pts, width, fill):
    """带圆角连接的粗折线(Pillow line joint=curve)。"""
    draw.line(pts, fill=fill, width=width, joint="curve")
    r = width // 2
    for (x, y) in (pts[0], pts[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def rotated_ellipse(draw, cx, cy, rx, ry, deg, fill):
    """旋转椭圆(音符头):多边形逼近,256 段足够平滑。"""
    a = math.radians(deg)
    n = 256
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x0, y0 = rx * math.cos(t), ry * math.sin(t)
        x = cx + x0 * math.cos(a) - y0 * math.sin(a)
        y = cy + x0 * math.sin(a) + y0 * math.cos(a)
        pts.append((x, y))
    draw.polygon(pts, fill=fill)


def draw_logo(size, dark=False):
    """按坐标生成 size×size 的 LOGO(先 4× 绘制再缩放)。"""
    k = CANVAS / 1024.0

    im = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    # 1. 圆角方底
    bg = DEEP if not dark else (16, 18, 26, 255)
    rrect(d, (0, 0, CANVAS - 1, CANVAS - 1), int(190 * k), bg)

    # 2. 八分音符(白)
    #    音符头:左下,旋转 -20°
    rotated_ellipse(d, 372 * k, 692 * k, 128 * k, 92 * k, -20, WHITE)
    #    符干:从符头右侧向上
    d.rounded_rectangle(
        (496 * k, 232 * k, 538 * k, 700 * k),
        radius=int(18 * k),
        fill=WHITE,
    )
    #    符尾 → 向上折线(声波):从符干顶部出发,锯齿上行
    zig = [
        (520, 268), (606, 312), (660, 214),
        (742, 258), (792, 158), (864, 200),
    ]
    thick_polyline(d, [(x * k, y * k) for (x, y) in zig], int(40 * k), WHITE)

    # 3. 右下角青点(AI 辅助)
    d.ellipse((768 * k, 768 * k, 906 * k, 906 * k), fill=CYAN)

    return im.resize((size, size), Image.LANCZOS)


def linear_gradient(w, h, top, bottom, horizontal=False):
    """线性渐变底图。"""
    base = Image.new("RGBA", (w, h))
    px = base.load()
    for i in range(w if horizontal else h):
        t = i / max(1, (w if horizontal else h) - 1)
        c = tuple(int(top[j] + (bottom[j] - top[j]) * t) for j in range(3)) + (255,)
        if horizontal:
            for y in range(h):
                px[i, y] = c
        else:
            for x in range(w):
                px[x, i] = c
    return base


def make_header(path):
    """NSIS 顶部横条 150×57:深蓝渐变 + 白色音符(小尺寸重绘,不用复杂折线)。"""
    w, h = 150, 57
    im = linear_gradient(w, h, (24, 34, 60), (30, 42, 74), horizontal=True)
    d = ImageDraw.Draw(im)
    # 白色八分音符(简化:符头+符干+小旗),整体居左
    d.ellipse((18, 34, 38, 48), fill=WHITE)                     # 符头
    d.rounded_rectangle((36, 8, 41, 42), radius=2, fill=WHITE)  # 符干
    d.line((40, 10, 52, 16, 50, 26), fill=WHITE, width=4, joint="curve")  # 旗
    d.ellipse((41, 44, 49, 52), fill=CYAN)                       # 青点
    d.text((62, 12), "Muno", fill=WHITE, font=load_font(22))
    im.convert("RGB").save(path, "BMP")
    print("header ->", path)


def make_sidebar(path):
    """NSIS 左侧竖幅 164×314:竖向深蓝渐变 + 大 LOGO + 品牌字。"""
    w, h = 164, 314
    im = linear_gradient(w, h, (22, 32, 58), (14, 18, 32))
    logo = draw_logo(112).resize((112, 112), Image.LANCZOS)
    im.alpha_composite(logo, ((w - 112) // 2, 52))
    d = ImageDraw.Draw(im)
    d.text((w // 2, 190), "Muno", fill=WHITE, anchor="mm", font=load_font(34))
    d.text((w // 2, 214), "AI 音乐工作站", fill=(160, 174, 196), anchor="mm", font=load_font(15, bold=False, cjk=True))
    # 底部荧光青装饰线
    d.line((28, 262, w - 28, 262), fill=CYAN, width=3)
    im.convert("RGB").save(path, "BMP")
    print("sidebar ->", path)


def main():
    icons = os.path.join(ROOT, "src-tauri", "icons")
    os.makedirs(icons, exist_ok=True)

    # PNG 系列
    logo512 = draw_logo(512)
    logo512.save(os.path.join(icons, "icon.png"))
    for size in (16, 24, 32, 48, 64, 128, 256):
        draw_logo(size).save(os.path.join(icons, f"{size}x{size}.png") if size not in (256,) else os.path.join(icons, "128x128@2x.png"))
    draw_logo(64).save(os.path.join(icons, "64x64.png"))
    draw_logo(32).save(os.path.join(icons, "32x32.png"))
    draw_logo(128).save(os.path.join(icons, "128x128.png"))
    draw_logo(256).save(os.path.join(icons, "128x128@2x.png"))

    # ICO 多尺寸
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    logo512.save(
        os.path.join(icons, "icon.ico"),
        format="ICO",
        sizes=ico_sizes,
    )
    print("icon.ico ->", os.path.join(icons, "icon.ico"))

    # ICNS(macOS,NSIS 打包不用但保持集合完整)
    try:
        logo512.save(os.path.join(icons, "icon.icns"), format="ICNS")
        print("icon.icns ->", os.path.join(icons, "icon.icns"))
    except Exception as e:  # noqa: BLE001 — icns 写失败不阻塞 Windows 交付
        print("icns skip:", e)

    # Windows 商店占位图(保持现有尺寸集合)
    store = {
        "Square30x30Logo.png": 30,
        "Square44x44Logo.png": 44,
        "Square71x71Logo.png": 71,
        "Square89x89Logo.png": 89,
        "Square107x107Logo.png": 107,
        "Square142x142Logo.png": 142,
        "Square150x150Logo.png": 150,
        "Square284x284Logo.png": 284,
        "Square310x310Logo.png": 310,
        "StoreLogo.png": 50,
    }
    for name, sz in store.items():
        draw_logo(sz).save(os.path.join(icons, name))
    print("store logos ->", icons)

    # favicon
    pub = os.path.join(ROOT, "public")
    logo512.save(os.path.join(pub, "icon.png"))
    print("public/icon.png ->", os.path.join(pub, "icon.png"))

    # NSIS 位图
    make_header(os.path.join(ROOT, "src-tauri", "installer", "header.bmp"))
    make_sidebar(os.path.join(ROOT, "src-tauri", "installer", "sidebar.bmp"))


if __name__ == "__main__":
    sys.exit(main())
