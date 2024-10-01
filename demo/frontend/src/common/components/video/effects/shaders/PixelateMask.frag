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

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform mediump vec2 uSize;
uniform lowp float uBlockSize;
uniform int uNumMasks;
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;

out vec4 fragColor;

void main() {
  vec4 color = texture(uSampler, vTexCoord);
  vec2 uv = vTexCoord.xy;
  float dx = uBlockSize / uSize.x;
  float dy = uBlockSize / uSize.y;

  vec4 color1 = vec4(0.0f);
  vec4 color2 = vec4(0.0f);
  vec4 color3 = vec4(0.0f);

  vec2 sampleCoord = (vec2(dx * floor((uv.x / dx)), dy * floor((uv.y / dy))) +
  vec2(dx * ceil((uv.x / dx)), dy * ceil((uv.y / dy)))) / 2.0f;
  vec4 frameColor = texture(uSampler, sampleCoord);
  color = frameColor;

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
    fragColor = color;
  } else {
    fragColor = vec4(0.0f);
  }
}