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
uniform vec2 uSize;
uniform int uBlurRadius;

out vec4 fragColor;

void main() {
  vec2 texOffset = 1.0f / uSize;
  // texel color
  vec3 color = texture(uSampler, vTexCoord).rgb;
  float sampleCount = 0.0f;

  // sample the surrounding pixels based on the blur radius
  for(int x = -uBlurRadius; x <= uBlurRadius; x++) {
    for(int y = -uBlurRadius; y <= uBlurRadius; y++) {
      vec2 offset = vec2(float(x), float(y)) * texOffset;
      color += texture(uSampler, vTexCoord + offset).rgb;
      sampleCount += 1.0f;
    }
  }

  // average the colors of the sampled pixels
  color /= sampleCount;
  fragColor = vec4(color, 1.0f);
}