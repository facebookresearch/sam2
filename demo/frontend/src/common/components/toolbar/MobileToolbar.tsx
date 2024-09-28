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
import MobileObjectsToolbar from '@/common/components/annotations/MobileObjectsToolbar';
import MobileEffectsToolbar from '@/common/components/effects/MobileEffectsToolbar';
import MoreOptionsToolbar from '@/common/components/options/MoreOptionsToolbar';

type Props = {
  tabIndex: number;
  onTabChange: (newIndex: number) => void;
};

export default function MobileToolbar({tabIndex, onTabChange}: Props) {
  const tabs = [
    <MobileObjectsToolbar key="objects" onTabChange={onTabChange} />,
    <MobileEffectsToolbar key="effects" onTabChange={onTabChange} />,
    <MoreOptionsToolbar key="more-options" onTabChange={onTabChange} />,
  ];

  return (
    <div className="relative flex flex-col bg-black">{tabs[tabIndex]}</div>
  );
}
