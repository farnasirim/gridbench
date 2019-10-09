import math


def lerp(start_point, end_point, value, maximum):
    return start_point + (end_point - start_point) * value / maximum

def lerp3(s, e, value, maximum):
    r1 = lerp(s[0], e[0], value, maximum)
    r2 = lerp(s[1], e[1], value, maximum)
    r3 = lerp(s[2], e[2], value, maximum)
    return (r1, r2, r3)

def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc= min(r, g, b)
    v = maxc
    if minc == maxc: return (0, 0, v)
    diff= maxc - minc
    s = diff / maxc
    rc = (maxc - r) / diff
    gc = (maxc - g) / diff
    bc = (maxc - b) / diff
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return (h, s, v)

def hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(math.floor(h*6.0))
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0 - f))
    if i % 6 == 0: return v, t, p
    if i % 6 == 1: return q, v, p
    if i % 6 == 2: return p, v, t
    if i % 6 == 3: return p, q, v
    if i % 6 == 4: return t, p, v
    if i % 6 == 5: return v, p, q

# start_triplet= rgb_to_hsv(0, 255, 0)
# end_triplet= rgb_to_hsv(255, 0, 0)
# 
# rgb_triplet_to_display= hsv_to_rgb(transition3(start_triplet, end_triplet, value, maximum))
