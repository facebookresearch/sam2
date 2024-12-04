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
import ObjectsToolbar from '@/common/components/annotations/ObjectsToolbar';
import EffectsToolbar from '@/common/components/effects/EffectsToolbar';
import MoreOptionsToolbar from '@/common/components/options/MoreOptionsToolbar';
import type {CSSProperties} from 'react';

type Props = {
  tabIndex: number;
  onTabChange: (newIndex: number) => void;
};

export default function DesktopToolbar({tabIndex, onTabChange}: Props) {
  const toolbarShadow: CSSProperties = {
    boxShadow: '0px 1px 3px 1px rgba(0,0,0,.25)',
    transition: 'box-shadow 0.8s ease-out',
  };

  const tabs = [
    <ObjectsToolbar key="objects" onTabChange={onTabChange} />,
    <EffectsToolbar key="effects" onTabChange={onTabChange} />,
    <MoreOptionsToolbar key="options" onTabChange={onTabChange} />,
  ];

  return (
    <div
      style={toolbarShadow}
      className="bg-graydark-800 text-white md:basis-[350px] lg:basis-[435px] shrink-0 rounded-xl">
      {tabs[tabIndex]}
    </div>
  );
}
