"""
Performs an image fitting test, outputting every combination of fit/anchor
"""
import os
import colorsys
import PIL
import PIL.Image
import PIL.ImageDraw

from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import fit_image, logger

TEST_SIZE=200

def main() -> None:
    with DebugUnifiedLoggingContext():
        black_rect_vert = PIL.Image.new("RGB", (TEST_SIZE // 4, TEST_SIZE // 2), (0, 0, 0))
        black_rect_horz = PIL.Image.new("RGB", (TEST_SIZE // 2, TEST_SIZE // 4), (0, 0, 0))
        
        grad_size = TEST_SIZE * 2

        grad_full = PIL.Image.new("RGB", (grad_size, grad_size))
        draw_full = PIL.ImageDraw.Draw(grad_full)
        grad_vert = PIL.Image.new("RGB", (TEST_SIZE, grad_size))
        draw_vert = PIL.ImageDraw.Draw(grad_vert)
        grad_horz = PIL.Image.new("RGB", (grad_size, TEST_SIZE))
        draw_horz = PIL.ImageDraw.Draw(grad_horz)

        for i in range(grad_size * 2):
            hue = (i + 1) / (grad_size * 2)
            rgb_float = colorsys.hsv_to_rgb(hue, 1.0, 0.7)
            r, g, b = [int(j * 255) for j in rgb_float]
            if i > grad_size:
                draw_full.line([(i % grad_size, grad_size), (grad_size, i % grad_size)], (r, g, b), width=1)
            else:
                half_hue = (i + 1) / grad_size
                half_rgb_float = colorsys.hsv_to_rgb(half_hue, 1.0, 0.7)
                h_r, h_g, h_b = [int(j * 255) for j in half_rgb_float]
                draw_horz.line([(i, 0), (i, grad_size)], (h_r, h_g, h_b), width=1)
                draw_vert.line([(0, i), (grad_size, i)], (h_r, h_g, h_b), width=1)
                draw_full.line([(i, 0), (0, i)], (r, g, b), width=1)

        for dirname, img in [
            ("vert", black_rect_vert),
            ("horz", black_rect_horz),
            ("grad-full", grad_full),
            ("grad-vert", grad_vert),
            ("grad-horz", grad_horz),
        ]:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "image-fit", dirname)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img.save(os.path.join(save_dir, "base.png"))
            for fit in ["actual", "stretch", "contain", "cover"]:
                for anchor in [
                    "top-left", "top-center", "top-right",
                    "center-left", "center-center", "center-right",
                    "bottom-left", "bottom-center", "bottom-right"
                ]:
                    save_path = os.path.join(save_dir, f"{fit}-{anchor}.png")
                    fit_image(img, TEST_SIZE, TEST_SIZE, fit=fit, anchor=anchor).save(save_path)
                    logger.info(f"Wrote {save_path}")

if __name__ == "__main__":
    main()
