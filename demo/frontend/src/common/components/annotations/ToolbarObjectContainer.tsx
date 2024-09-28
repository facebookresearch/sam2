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
import {spacing} from '@/theme/tokens.stylex';
import {Close} from '@carbon/icons-react';
import stylex from '@stylexjs/stylex';
import {PropsWithChildren, ReactNode} from 'react';

const sharedStyles = stylex.create({
  container: {
    display: 'flex',
    overflow: 'hidden',
    cursor: 'pointer',
    flexShrink: 0,
    borderTop: 'none',
    backgroundColor: {
      '@media screen and (max-width: 768px)': '#000',
    },
    paddingHorizontal: {
      default: spacing[8],
      '@media screen and (max-width: 768px)': spacing[5],
    },
    paddingBottom: {
      default: spacing[8],
      '@media screen and (max-width: 768px)': 10,
    },
  },
  activeContainer: {
    background: '#000',
    borderRadius: 16,
    marginHorizontal: 16,
    padding: {
      default: spacing[4],
      '@media screen and (max-width: 768px)': spacing[5],
    },
    marginBottom: {
      default: spacing[8],
      '@media screen and (max-width: 768px)': 0,
    },
  },
  itemsCenter: {
    alignItems: 'center',
  },
  rightColumn: {
    marginStart: {
      default: spacing[4],
      '@media screen and (max-width: 768px)': 0,
    },
    flexGrow: 1,
    alignItems: 'center',
  },
});

type ToolbarObjectContainerProps = PropsWithChildren<{
  alignItems?: 'top' | 'center';
  isActive: boolean;
  title: string;
  subtitle: string;
  thumbnail: ReactNode;
  isMobile: boolean;
  onCancel?: () => void;
  onClick?: () => void;
}>;

export default function ToolbarObjectContainer({
  alignItems = 'top',
  children,
  isActive,
  title,
  subtitle,
  thumbnail,
  isMobile,
  onClick,
  onCancel,
}: ToolbarObjectContainerProps) {
  if (isMobile) {
    return (
      <div
        onClick={onClick}
        {...stylex.props(sharedStyles.container, sharedStyles.itemsCenter)}>
        <div {...stylex.props(sharedStyles.rightColumn)}>{children}</div>
      </div>
    );
  }

  return (
    <div
      onClick={onClick}
      {...stylex.props(
        sharedStyles.container,
        isActive && sharedStyles.activeContainer,
        alignItems === 'center' && sharedStyles.itemsCenter,
      )}>
      {thumbnail}
      <div {...stylex.props(sharedStyles.rightColumn)}>
        <div className="text-md font-semibold ml-2">{title}</div>
        {subtitle.length > 0 && (
          <div className="text-sm text-gray-400 leading-5 mt-2 ml-2">
            {subtitle}
          </div>
        )}
        {children}
      </div>
      {onCancel != null && (
        <div className="items-start self-stretch" onClick={onCancel}>
          <Close size={32} />
        </div>
      )}
    </div>
  );
}
