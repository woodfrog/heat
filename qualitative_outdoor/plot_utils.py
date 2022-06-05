import cv2
import svgwrite
import colorsys


def plot_preds(image, corners, edges):
    for line in edges:
        cv2.line(image, tuple(line[:2]), tuple(line[2:]), (255, 255, 0), 2)
    for c in corners:
        cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
    return image


def random_colors(N, bright=True, same=False, colors=None):
    brightness = 1.0 if bright else 0.7
    if colors is None or same:
        if same:
            hsv = [(0, 1, brightness) for i in range(N)]
        else:
            hsv = [(i / N, 1, brightness) for i in range(N)]
    else:
        hsv = [(colors[i], 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def svg_generate(image_link, corners, edges, name, size=512):
    dwg = svgwrite.Drawing(name + '.svg', size=('{}'.format(size), '{}'.format(size)))
    shapes = dwg.add(dwg.g(id='shape', fill='black'))
    # colors = random_colors(len(edges), same=True)
    shapes.add(dwg.image(href=image_link, size=(size, size)))

    scale = size / 256
    for i, edge in enumerate(edges):
        x = edge[:2] * scale
        y = edge[2:] * scale
        shapes.add(dwg.line((int(x[0]), int(x[1])), (int(y[0]), int(y[1])),
                            stroke="#EE6507", stroke_width=3*scale, opacity=0.7))

    for i, corner in enumerate(corners):
        shapes.add(dwg.circle((int(corners[i][0] * scale), int(corners[i][1]) * scale), r=4*scale,
                              stroke='green', fill='white', stroke_width=2*scale, opacity=0.8))
    return dwg
