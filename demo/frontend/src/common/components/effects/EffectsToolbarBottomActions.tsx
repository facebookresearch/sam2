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
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';
import RestartSessionButton from '@/common/components/session/RestartSessionButton';
import ToolbarBottomActionsWrapper from '@/common/components/toolbar/ToolbarBottomActionsWrapper';
import {
  MORE_OPTIONS_TOOLBAR_INDEX,
  OBJECT_TOOLBAR_INDEX,
} from '@/common/components/toolbar/ToolbarConfig';
import {ChevronRight} from '@carbon/icons-react';

type Props = {
  onTabChange: (newIndex: number) => void;
};

export default function EffectsToolbarBottomActions({onTabChange}: Props) {
  function handleSwitchToMoreOptionsTab() {
    onTabChange(MORE_OPTIONS_TOOLBAR_INDEX);
  }

  return (
    <ToolbarBottomActionsWrapper>
      <RestartSessionButton
        onRestartSession={() => onTabChange(OBJECT_TOOLBAR_INDEX)}
      />
      <PrimaryCTAButton
        onClick={handleSwitchToMoreOptionsTab}
        endIcon={<ChevronRight />}>
        Next
      </PrimaryCTAButton>
    </ToolbarBottomActionsWrapper>
  );
}
