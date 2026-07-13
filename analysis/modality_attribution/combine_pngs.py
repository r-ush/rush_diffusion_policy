#!/usr/bin/env python
"""여러 PNG를 하나로 합쳐 한 화면에서 같이 보게 한다 (세로/가로 스택).

attribution 그림들(예: vision_saliency.png + attribution_timeline.png)을 한 파일로 묶어
같이 보려는 용도. 라벨(파일 basename)을 각 이미지 위에 얹는다.

사용 예:
  python -m analysis.modality_attribution.combine_pngs \
      data/.../attribution_ep013/detail/vision_saliency.png \
      data/.../attribution_ep013/attribution_timeline.png \
      -o data/.../attribution_ep013/combined.png
"""
from __future__ import annotations

import pathlib

import click
from PIL import Image, ImageDraw, ImageFont


def _label_height(layout):
    return 34 if layout == "vertical" else 0


def _draw_label(img, text):
    """이미지 위에 얇은 라벨 바를 붙여 반환."""
    bar_h = 34
    out = Image.new("RGB", (img.size[0], img.size[1] + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.size[0], bar_h], fill=(30, 30, 30))
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return out


@click.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("-o", "--output", required=True, help="합쳐 저장할 PNG 경로")
@click.option("--layout", type=click.Choice(["vertical", "horizontal"]), default="vertical",
              show_default=True)
@click.option("--size", default=2000, type=int, show_default=True,
              help="vertical=공통 폭(px), horizontal=공통 높이(px)")
@click.option("--label/--no-label", default=True, help="각 이미지에 파일명 라벨 붙이기")
def main(inputs, output, layout, size, label):
    imgs = []
    for p in inputs:
        im = Image.open(p).convert("RGB")
        if label:
            im = _draw_label(im, pathlib.Path(p).name)
        imgs.append(im)

    if layout == "vertical":
        resized = []
        for im in imgs:
            w, h = im.size
            resized.append(im.resize((size, max(1, int(h * size / w)))))
        total_h = sum(im.size[1] for im in resized)
        canvas = Image.new("RGB", (size, total_h), (255, 255, 255))
        y = 0
        for im in resized:
            canvas.paste(im, (0, y))
            y += im.size[1]
    else:
        resized = []
        for im in imgs:
            w, h = im.size
            resized.append(im.resize((max(1, int(w * size / h)), size)))
        total_w = sum(im.size[0] for im in resized)
        canvas = Image.new("RGB", (total_w, size), (255, 255, 255))
        x = 0
        for im in resized:
            canvas.paste(im, (x, 0))
            x += im.size[0]

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Combined image saved: {out}  ({canvas.size[0]}x{canvas.size[1]}, {len(imgs)} panels)")


if __name__ == "__main__":
    main()
