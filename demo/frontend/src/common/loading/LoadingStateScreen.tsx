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
import introVideo from '@/assets/videos/sam2_720px_dark.mp4';
import introVideoPoster from '@/assets/videos/sam2_video_poster.png';
import StaticVideoPlayer from '@/common/loading/StaticVideoPlayer';
import {borderRadius, fontSize, spacing} from '@/theme/tokens.stylex';
import stylex from '@stylexjs/stylex';
import {PropsWithChildren, ReactNode} from 'react';
import {Link} from 'react-router-dom';

const styles = stylex.create({
  container: {
    backgroundColor: '#000',
    minHeight: '100%',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: spacing[8],
    maxWidth: '36rem', //* 576px */
    marginHorizontal: 'auto',
    paddingVertical: {
      default: '6rem',
      '@media screen and (max-width: 768px)': '3rem',
    },
    paddingHorizontal: spacing[8],
    color: '#fff',
  },
  animationContainer: {
    display: 'flex',
    justifyContent: 'center',
  },
  animation: {
    border: '2px solid white',
    borderRadius: borderRadius['xl'],
    maxWidth: 450,
    maxHeight: 450,
    height: '100%',
    overflow: 'hidden',
    '@media screen and (max-width: 768px)': {
      height: 300,
      width: 300,
    },
  },
  title: {
    textAlign: 'center',
    lineHeight: '2rem',
    fontSize: fontSize['2xl'],
    fontWeight: 400,
  },
  description: {
    textAlign: 'center',
    color: '#A7B3BF',
  },
  link: {
    textAlign: 'center',
    textDecorationLine: 'underline',
    color: '#A7B3BF',
  },
});

type Props = PropsWithChildren<{
  title: string;
  description?: string | ReactNode;
  linkProps?: {
    to: string;
    label: string;
  };
}>;

export default function LoadingStateScreen({
  title,
  description,
  children,
  linkProps,
}: Props) {
  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.content)}>
        <div {...stylex.props(styles.animationContainer)}>
          <div {...stylex.props(styles.animation)}>
            <StaticVideoPlayer
              src={introVideo}
              aspectRatio="square"
              poster={introVideoPoster}
              muted={true}
              loop={true}
              autoPlay={true}
              playsInline={true}
              controls={false}
            />
          </div>
        </div>
        <h2 {...stylex.props(styles.title)}>{title}</h2>
        {description != null && (
          <div {...stylex.props(styles.description)}>{description}</div>
        )}
        {children}
        {linkProps != null && (
          <Link to={linkProps.to} {...stylex.props(styles.link)}>
            {linkProps.label}
          </Link>
        )}
      </div>
    </div>
  );
}
