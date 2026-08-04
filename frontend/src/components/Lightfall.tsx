import { useEffect, useRef } from "react";
import { Mesh, Program, Renderer, Triangle } from "ogl";

import "./Lightfall.css";

const MAX_COLORS = 8;

function hexToRGB(hex: string): [number, number, number] {
  const value = hex.replace("#", "").padEnd(6, "0");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16) / 255) as [number, number, number];
}

function prepColors(input: string[]) {
  const base = (input.length ? input : ["#9ae6b4", "#28a96b", "#67e8f9"]).slice(0, MAX_COLORS);
  const colors = Array.from({ length: MAX_COLORS }, (_, index) => hexToRGB(base[Math.min(index, base.length - 1)]));
  const average = base.reduce(
    (sum, color) => hexToRGB(color).map((channel, index) => sum[index] + channel) as [number, number, number],
    [0, 0, 0] as [number, number, number],
  ).map((channel) => channel / base.length) as [number, number, number];
  return { colors, count: base.length, average };
}

const vertex = `
attribute vec2 position;
attribute vec2 uv;
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position, 0.0, 1.0); }
`;

const fragment = `
precision highp float;
uniform vec3 iResolution;
uniform vec2 iMouse;
uniform float iTime;
uniform vec3 uColor0; uniform vec3 uColor1; uniform vec3 uColor2; uniform vec3 uColor3;
uniform vec3 uColor4; uniform vec3 uColor5; uniform vec3 uColor6; uniform vec3 uColor7;
uniform int uColorCount;
uniform vec3 uBgColor; uniform vec3 uMouseColor;
uniform float uSpeed; uniform int uStreakCount; uniform float uStreakWidth;
uniform float uStreakLength; uniform float uGlow; uniform float uDensity;
uniform float uTwinkle; uniform float uZoom; uniform float uBgGlow;
uniform float uOpacity; uniform float uMouseEnabled; uniform float uMouseStrength;
uniform float uMouseRadius;
varying vec2 vUv;

vec3 palette(float h) {
  int count = uColorCount;
  if (count < 1) count = 1;
  int idx = int(floor(clamp(h, 0.0, 0.999999) * float(count)));
  if (idx <= 0) return uColor0; if (idx == 1) return uColor1;
  if (idx == 2) return uColor2; if (idx == 3) return uColor3;
  if (idx == 4) return uColor4; if (idx == 5) return uColor5;
  if (idx == 6) return uColor6; return uColor7;
}

vec3 tanhv(vec3 x) { vec3 e = exp(-2.0 * x); return (1.0 - e) / (1.0 + e); }

vec2 sceneC(vec2 frag, vec2 r) {
  vec2 p = (frag + frag - r) / r.x;
  float z = 0.0; float d = 1e3; vec4 o = vec4(0.0);
  for (int k = 0; k < 39; k++) {
    if (d <= 1e-4) break;
    o = z * normalize(vec4(p, uZoom, 0.0)) - vec4(0.0, 4.0, 1.0, 0.0) / 4.5;
    d = 1.0 - sqrt(length(o * o)); z += d;
  }
  return vec2(o.x, atan(o.z, o.y));
}

void mainImage(out vec4 outputColor, vec2 c) {
  vec2 r = iResolution.xy;
  vec2 uv0 = (c + c - r) / r.x;
  float time = 0.1 * iTime * uSpeed + 9.0;
  float rings = max(1.0, floor(6.28318530718 * max(uDensity, 0.05) + 0.5));
  vec2 cell = vec2(5e-3, 6.28318530718 / rings);
  vec2 c0 = sceneC(c, r); vec2 cdx = sceneC(c + vec2(1.0, 0.0), r); vec2 cdy = sceneC(c + vec2(0.0, 1.0), r);
  vec2 dCx = cdx - c0; vec2 dCy = cdy - c0;
  dCx.y -= 6.28318530718 * floor(dCx.y / 6.28318530718 + 0.5);
  dCy.y -= 6.28318530718 * floor(dCy.y / 6.28318530718 + 0.5);
  vec2 fw = abs(dCx) + abs(dCy); c = c0;
  vec2 p = vec2(2.0, 1.0) * uv0 - (r / r.x) * vec2(0.0, 1.0);
  vec4 glow = vec4(uBgColor * 90.0 * uBgGlow / (1e3 * dot(p, p) + 6.0), 0.0);
  float mouseGlow = 0.0;
  if (uMouseEnabled > 0.5) {
    vec2 mouse = (iMouse + iMouse - r) / r.x;
    float distanceToMouse = length(uv0 - mouse);
    mouseGlow = exp(-distanceToMouse * distanceToMouse / max(uMouseRadius * uMouseRadius, 1e-4)) * uMouseStrength;
    glow.rgb += uMouseColor * mouseGlow * 0.25;
  }
  float radius = 5e-4 * uStreakWidth;
  vec2 feather = vec2(max(length(fw), 1e-5));
  float tail = 19.0 / max(uStreakLength, 0.05);
  for (int m = 0; m < 16; m++) {
    if (m >= uStreakCount) break;
    float layer = float(m) + 1.0;
    float noise = fract(sin(dot(vec2(layer, floor(c.x / cell.x + 0.5)), vec2(7.0, 11.0)) * 73.0));
    vec2 point = c - (time + time * noise) * vec2(0.0, 1.0);
    point -= floor(point / cell + 0.5) * cell;
    float hue = fract(8663.0 * noise);
    vec3 color = palette(hue);
    float weight = mix(1.5, 1.0 + sin(time + 7.0 * hue + 4.0), uTwinkle) * (1.0 + mouseGlow * 2.0);
    vec2 inner = vec2(length(max(point, vec2(-1.0, 0.0))), length(point) - radius) - radius;
    vec2 soft = vec2(1.0) - smoothstep(-feather, feather, inner);
    glow.rgb += dot(soft, vec2(exp(tail * point.y), 3.0)) * color * weight;
    c.x += cell.x / 8.0;
  }
  vec3 color = sqrt(tanhv(max(glow.rgb * uGlow - vec3(0.04, 0.08, 0.02), 0.0)));
  outputColor = vec4(color, uOpacity);
}

void main() { vec4 color; mainImage(color, vUv * iResolution.xy); gl_FragColor = color; }
`;

type LightfallProps = {
  colors: string[];
  backgroundColor: string;
  paused?: boolean;
  className?: string;
};

export default function Lightfall({ colors, backgroundColor, paused = false, className = "" }: LightfallProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const renderer = new Renderer({ dpr: Math.min(window.devicePixelRatio || 1, 1.5), alpha: true, antialias: true });
    const gl = renderer.gl;
    const canvas = gl.canvas;
    Object.assign(canvas.style, { width: "100%", height: "100%", display: "block" });
    container.appendChild(canvas);
    const prepared = prepColors(colors);
    const uniforms = {
      iResolution: { value: [gl.drawingBufferWidth, gl.drawingBufferHeight, 1] },
      iMouse: { value: [0, 0] }, iTime: { value: 0 },
      uColor0: { value: prepared.colors[0] }, uColor1: { value: prepared.colors[1] },
      uColor2: { value: prepared.colors[2] }, uColor3: { value: prepared.colors[3] },
      uColor4: { value: prepared.colors[4] }, uColor5: { value: prepared.colors[5] },
      uColor6: { value: prepared.colors[6] }, uColor7: { value: prepared.colors[7] },
      uColorCount: { value: prepared.count }, uBgColor: { value: hexToRGB(backgroundColor) },
      uMouseColor: { value: prepared.average }, uSpeed: { value: 0.32 },
      uStreakCount: { value: 3 }, uStreakWidth: { value: 1 }, uStreakLength: { value: 0.7 },
      uGlow: { value: 0.86 }, uDensity: { value: 0.85 }, uTwinkle: { value: 0.7 },
      uZoom: { value: 2.45 }, uBgGlow: { value: 0.8 }, uOpacity: { value: 0.9 },
      uMouseEnabled: { value: paused ? 0 : 1 }, uMouseStrength: { value: 0.34 }, uMouseRadius: { value: 0.76 },
    };
    const program = new Program(gl, { vertex, fragment, uniforms });
    const geometry = new Triangle(gl);
    const mesh = new Mesh(gl, { geometry, program });
    const resize = () => {
      const rect = container.getBoundingClientRect();
      renderer.setSize(Math.max(rect.width, 1), Math.max(rect.height, 1));
      uniforms.iResolution.value = [gl.drawingBufferWidth, gl.drawingBufferHeight, 1];
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = renderer.dpr || 1;
      uniforms.iMouse.value = [(event.clientX - rect.left) * dpr, (rect.height - event.clientY + rect.top) * dpr];
    };
    if (!paused) canvas.addEventListener("pointermove", onPointerMove);
    let frame = 0;
    const loop = (time: number) => {
      uniforms.iTime.value = paused ? 0 : time * 0.001;
      renderer.render({ scene: mesh });
      if (!paused) frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(frame);
      canvas.removeEventListener("pointermove", onPointerMove);
      observer.disconnect();
      canvas.remove();
    };
  }, [backgroundColor, colors, paused]);

  return <div ref={containerRef} className={`lightfall-container ${className}`} aria-hidden="true" />;
}
