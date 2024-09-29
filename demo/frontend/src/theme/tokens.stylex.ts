/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import * as stylex from '@stylexjs/stylex';

export const spacing = stylex.defineVars({
  '0': '0rem',
  '0.5': '0.125rem',
  '1': '0.25rem',
  '1.5': '0.375rem',
  '2': '0.5rem',
  '2.5': '0.625rem',
  '3': '0.75rem',
  '3.5': '0.875rem',
  '4': '1rem',
  '5': '1.25rem',
  '6': '1.5rem',
  '7': '1.75rem',
  '8': '2rem',
  '9': '2.25rem',
  '10': '2.5rem',
  '11': '2.75rem',
  '12': '3rem',
});

export const gap = stylex.defineVars({
  4: '1rem' /* 16px */,
});

export const w = stylex.defineVars({
  full: '100%',
  12: '3rem' /* 48px */,
  96: '24rem' /* 384px */,
});

export const m = stylex.defineVars({
  3: '0.75rem' /* 12px */,
});

export const fontSize = stylex.defineVars({
  xs: '0.75rem',
  sm: '0.875rem',
  base: '1rem',
  lg: '1.125rem',
  xl: '1.25rem',
  '2xl': '1.5rem',
});

export const fontWeight = stylex.defineVars({
  thin: 100,
  extralight: 200,
  light: 300,
  normal: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
  extrabold: 800,
});

export const color = stylex.defineVars({
  subtitle: 'rgb(107 114 128)',
  'gray-900': 'rgb(17 24 39)',
  'gray-800': 'rgb(26 28 31)',
  'gray-700': 'rgb(55 62 65)',
  'blue-600': 'rgb(37 99 235)',
});

export const screenSizes = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
};

export const borderRadius = stylex.defineVars({
  sm: '0.125rem',
  md: '0.375rem',
  lg: '0.5rem',
  xl: '0.75rem',
});

export const top = stylex.defineVars({
  0: 0,
  1: '0.25rem' /* 4px */,
  2: '0.5rem' /* 8px */,
});

export const right = stylex.defineVars({
  0: 0,
  1: '0.25rem' /* 4px */,
  2: '0.5rem' /* 8px */,
});

export const gradients = stylex.defineVars({
  rainbow:
    'linear-gradient(#000, #000) padding-box, linear-gradient(to right bottom, #FB73A5,#595FEF,#94EAE2,#FCCB6B) border-box',

  rainbowReverse:
    'linear-gradient(#000, #000) padding-box, linear-gradient(to left top, #FB73A5,#595FEF,#94EAE2,#FCCB6B) border-box',

  yellowTeal:
    'linear-gradient(#000, #000) padding-box, linear-gradient(to right bottom, #94EAE2,#FCCB6B) border-box',
});
