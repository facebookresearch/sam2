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
import tailwindCSSTypography from '@tailwindcss/typography';
import daisyui from 'daisyui';
import * as daisyColorThemes from 'daisyui/src/theming/themes';

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
    'node_modules/daisyui/dist/**/*.js',
    'node_modules/react-daisyui/dist/**/*.js',
  ],

  daisyui: {
    styled: true,
    themes: [
      {
        light: {
          ...daisyColorThemes['[data-theme=light]'],
          'base-100': '#FFFFFF',
          'base-200': '#F1F4F7',
          'base-300': '#DEE3E9',
          primary: '#0064E0',
          'primary-content': '#FFFFFF',
          secondary: '#0F191E',
          'secondary-content': '#FFFFFF',
          accent: '#6441D2',
          'accent-content': '#FFFFFF',
          info: '#009B9B',
          'info-content': '#FFFFFF',
          success: '#0F9B14',
          'success-content': '#FFFFFF',
          warning: '#FA8719',
          'warning-content': '#FFFFFF',
          error: '#C80A28',
          'error-content': '#FFFFFF',

          '--rounded-box': '0.35rem', // border radius rounded-box utility class, used in card and other large boxes
          '--rounded-btn': '0.35rem', // border radius rounded-btn utility class, used in buttons and similar element
          '--rounded-badge': '1rem', // border radius rounded-badge utility class, used in badges and similar
        },
      },
      'dark',
    ],
  },
  theme: {
    fontSize: {
      xs: ['0.75rem', {lineHeight: '1.5'}],
      sm: ['0.875rem', {lineHeight: '1.5'}],
      base: ['1rem', {lineHeight: '1.5'}],
      lg: ['1.125rem', {lineHeight: '1.2', fontWeight: 500}],
      xl: ['1.25rem', {lineHeight: '1.2', fontWeight: 500}],
      '2xl': [
        '1.5rem',
        {lineHeight: '1.2', fontWeight: 500, letterSpacing: '0.005rem'},
      ],
      '3xl': [
        '2.25rem',
        {lineHeight: '1.2', fontWeight: 500, letterSpacing: '0.01rem'},
      ],
      '4xl': [
        '3rem',
        {lineHeight: '1.2', fontWeight: 500, letterSpacing: '0.016rem'},
      ],
      '5xl': [
        '4rem',
        {lineHeight: '1.2', fontWeight: 400, letterSpacing: '0.016rem'},
      ],
      '6xl': [
        '5rem',
        {lineHeight: '1.2', fontWeight: 400, letterSpacing: '0.016rem'},
      ],
    },
    extend: {
      colors: {
        graydark: {
          50: '#f1f4f7',
          100: '#DEE3E9',
          200: '#CBD2D9',
          300: '#A7B3BF',
          400: '#8595A4',
          500: '#667788',
          600: '#465A69',
          700: '#343845',
          800: '#1A1C1F',
          900: '#0F191E',
        },
      },
      lineHeight: {
        tight: 1.2,
      },
      backgroundImage: {
        dot: 'url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAAXNSR0IArs4c6QAAABdJREFUGBljYGBg+A/FQAoTMGEKDUcRAATwAgFGIXEOAAAAAElFTkSuQmCC)',
      },
      keyframes: {
        wiggle: {
          '0%, 100%': {transform: 'rotate(-3deg)'},
          '50%': {transform: 'rotate(3deg)'},
        },
      },
      animation: {
        wiggle: 'wiggle .25s ease-in-out',
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: '100%', // add required value here
            a: {
              textDecoration: 'none',
            },
          },
        },
      },
    },
  },
  plugins: [tailwindCSSTypography, daisyui],
};
