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

precision mediump float;
precision mediump sampler3D;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform float uCurrentFrame;
uniform sampler3D uColorGradeLUT;
uniform int uNumMasks;
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;

out vec4 fragColor;

void main() {
  vec4 color = texture(uSampler, vTexCoord);
  vec3 gradedColor = texture(uColorGradeLUT, color.rgb).rgb;

  vec4 color1 = vec4(0.0f);
  vec4 color2 = vec4(0.0f);
  vec4 color3 = vec4(0.0f);

  // Apply edge detection for each mask
  // We can't use dynamic indexing with samplers in GLSL ES 3.0.
  // https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf Ch 12.30
  if(uNumMasks > 0) {
    color1 = texture(uMaskTexture0, vec2(vTexCoord.y, vTexCoord.x));
  }
  if(uNumMasks > 1) {
    color2 = texture(uMaskTexture1, vec2(vTexCoord.y, vTexCoord.x));
  }
  if(uNumMasks > 2) {
    color3 = texture(uMaskTexture2, vec2(vTexCoord.y, vTexCoord.x));
  }

  bool overlap = (color1.r > 0.0f || color2.r > 0.0f || color3.r > 0.0f);
  if(overlap) {
    fragColor = vec4(gradedColor, 1);
  } else {
    fragColor = vec4(0.0f);
  }
}