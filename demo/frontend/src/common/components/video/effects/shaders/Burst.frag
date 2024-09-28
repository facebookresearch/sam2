#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision highp float;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize; // resolution
uniform int uNumMasks;
uniform bool uLineColor;
uniform bool uInterleave;
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;

uniform vec4 uMaskColor0;
uniform vec4 uMaskColor1;
uniform vec4 uMaskColor2;

uniform vec4 bbox0;
uniform vec4 bbox1;
uniform vec4 bbox2;

out vec4 fragColor;

void main() {
  float PI = radians(180.0f);
  float lines = uInterleave ? 12.0f : 80.0f;
  vec4 color = texture(uSampler, vTexCoord);
  vec4 color1 = uMaskColor0 / 255.0;
  vec4 color2 = uMaskColor1 / 255.0;
  vec4 color3 = uMaskColor2 / 255.0;

  vec4 mask1 = vec4(0.0f);
  vec4 mask2 = vec4(0.0f);
  vec4 mask3 = vec4(0.0f);
  vec4 scopedColor = vec4(0.0f);

  vec2 fragCoord = vTexCoord * uSize; // transform to pixel space
  bool scoped = false;
  vec4 transparent = vec4(0.0);
  float p = PI / lines;

  if(uNumMasks > 0) {
    mask1 = texture(uMaskTexture0, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center1 = (bbox0.xy + bbox0.zw) * 0.5f * uSize;
    vec2 fragCoordT = (fragCoord - center1) / uSize.y;
    float a = mod(atan(fragCoordT.y, fragCoordT.x) + p, p + p) - p; // angle of fragment

    float pattern = sin(a * lines);
    // smoothstep for antialiasing
    float line = smoothstep(2.8 / uSize.y, 0.0, length(fragCoordT) * abs(sin(a)));
    
    vec4 colorToBlend = uLineColor ? vec4(color1.rgb, 0.80f) : vec4(1.0f);
    bool visible = bbox0 != vec4(0.0f);

    if (uInterleave && visible) {
      vec4 tempColor = mix(transparent, colorToBlend, step(0.0, pattern));
      scopedColor += tempColor;
      scoped = true;
    } else if (!uInterleave && visible) {
      vec4 tempColor = uLineColor ? vec4(color1.rgb * line, line) : vec4(line);
      scopedColor += tempColor;
      scoped = true;
    }
  }

  if(uNumMasks > 1) {
    mask2 = texture(uMaskTexture1, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center2 = (bbox1.xy + bbox1.zw) * 0.5f * uSize;
    vec2 fragCoordT = (fragCoord - center2) / uSize.y;
    float a = mod(atan(fragCoordT.y, fragCoordT.x) + p, p + p) - p; // angle of fragment

    float pattern = sin(a * lines);
    float line = smoothstep(2.8 / uSize.y, 0.0, length(fragCoordT) * abs(sin(a)));
    
    vec4 colorToBlend = uLineColor ? vec4(color2.rgb, 0.8f) : vec4(1.0f);
    bool visible = bbox1 != vec4(0.0f);

    if (uInterleave && visible) {
      vec4 tempColor = mix(transparent, colorToBlend, step(0.0, pattern));
      if (scopedColor == vec4(0.0)) {
        scopedColor += tempColor;
      }
      scoped = true;
    } else if (!uInterleave && visible) {
      vec4 tempColor = uLineColor ? vec4(color2.rgb * line, line) : vec4(line);
      scopedColor += tempColor;    
      scoped = true;
    }
  }

  if (uNumMasks > 2) {
    mask3 = texture(uMaskTexture2, vec2(vTexCoord.y, vTexCoord.x));

    vec2 center3 = (bbox2.xy + bbox2.zw) * 0.5f * uSize;
    vec2 fragCoordT = (fragCoord - center3) / uSize.y;

    float a = mod(atan(fragCoordT.y, fragCoordT.x) + p, p + p) - p; // angle of fragment

    float pattern = sin(a * lines);
    float line = smoothstep(2.8 / uSize.y, 0.0, length(fragCoordT) * abs(sin(a)));

    vec4 colorToBlend = uLineColor ? vec4(color3.rgb, 0.8f) : vec4(1.0f);
    bool visible = bbox2 != vec4(0.0f);

    if (uInterleave && visible) {
      vec4 tempColor = mix(transparent, colorToBlend, step(0.0, pattern));
      if (scopedColor == vec4(0.0)) {
        scopedColor += tempColor;
      }
      scoped = true;
    } else if (!uInterleave && visible) {
      vec4 tempColor = uLineColor ? vec4(color3.rgb * line, line) : vec4(line);
      scopedColor += tempColor;
      scoped = true;
    }
  }

  bool overlap = (mask1.r > 0.0f || mask2.r > 0.0f || mask3.r > 0.0f);
  if(scoped) {
    fragColor = overlap ? color : scopedColor;
  } else {
    fragColor = overlap ? color : vec4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}
