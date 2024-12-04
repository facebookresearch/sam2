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
import GradientBorder from '@/common/components/button/GradientBorder';
import useScreenSize from '@/common/screen/useScreenSize';
import {BLUE_PINK_FILL_BR} from '@/theme/gradientStyle';
import type {CarbonIconType} from '@carbon/icons-react';
import {Loading} from 'react-daisyui';

type Props = {
  isDisabled?: boolean;
  isActive?: boolean;
  icon: CarbonIconType;
  title: string;
  badge?: React.ReactNode;
  variant: 'toggle' | 'button' | 'gradient' | 'flat';
  span?: 1 | 2;
  loadingProps?: {
    loading: boolean;
    label?: string;
  };
  onClick: () => void;
};

export default function ToolbarActionIcon({
  variant,
  isDisabled = false,
  isActive = false,
  title,
  badge,
  loadingProps,
  icon: Icon,
  span = 1,
  onClick,
}: Props) {
  const {isMobile} = useScreenSize();
  const isLoading = loadingProps?.loading === true;

  function handleClick() {
    if (isDisabled) {
      return;
    }
    onClick();
  }

  const ButtonBase = (
    <div
      onClick={handleClick}
      className={`relative rounded-lg h-full flex items-center justify-center select-none
      ${!isDisabled && 'cursor-pointer hover:bg-black'}
      ${span === 1 && 'col-span-1'} 
      ${span === 2 && 'col-span-2'}
      ${variant === 'button' && (isDisabled ? 'bg-graydark-500 text-gray-300' : 'bg-graydark-700 hover:bg-graydark-800 text-white')} 
      ${variant === 'toggle' && (isActive ? BLUE_PINK_FILL_BR : 'bg-inherit')}
      ${variant === 'flat' && (isDisabled ? ' text-gray-600' : 'text-white')} 
      `}>
      <div className="py-4 px-2">
        <div className="flex items-center justify-center">
          {isLoading ? (
            <Loading size="md" className="mx-auto" />
          ) : (
            <Icon
              size={isMobile ? 24 : 28}
              color={isActive ? 'white' : 'black'}
              className={`mx-auto ${isDisabled ? 'text-gray-300' : 'text-white'}`}
            />
          )}
        </div>
        <div
          className={`mt-1 md:mt-2 text-center text-xs font-bold ${isActive && 'text-white'}`}>
          {isLoading && loadingProps?.label != null
            ? loadingProps.label
            : title}
        </div>
        {isActive && badge}
      </div>
    </div>
  );

  return variant == 'gradient' ? (
    <GradientBorder rounded={false} className="rounded-lg h-full text-white">
      {ButtonBase}
    </GradientBorder>
  ) : (
    ButtonBase
  );
}
