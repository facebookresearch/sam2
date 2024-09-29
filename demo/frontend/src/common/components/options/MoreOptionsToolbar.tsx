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
import MoreOptionsToolbarBottomActions from '@/common/components/options/MoreOptionsToolbarBottomActions';
import ShareSection from '@/common/components/options/ShareSection';
import TryAnotherVideoSection from '@/common/components/options/TryAnotherVideoSection';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import ToolbarHeaderWrapper from '@/common/components/toolbar/ToolbarHeaderWrapper';
import useScreenSize from '@/common/screen/useScreenSize';
import {useEffect, useRef} from 'react';

type Props = {
  onTabChange: (newIndex: number) => void;
};

export default function MoreOptionsToolbar({onTabChange}: Props) {
  const {isMobile} = useScreenSize();
  const {clearMessage} = useMessagesSnackbar();
  const didClearMessageSnackbar = useRef(false);

  useEffect(() => {
    if (!didClearMessageSnackbar.current) {
      didClearMessageSnackbar.current = true;
      clearMessage();
    }
  }, [clearMessage]);

  return (
    <div className="flex flex-col h-full">
      <div className="grow">
        <ToolbarHeaderWrapper
          title="Nice work! What's next?"
          className="pb-0 !border-b-0 !text-white"
          showProgressChip={false}
        />
        <ShareSection />
        {!isMobile && <div className="h-[1px] bg-black mt-4 mb-8"></div>}
        <TryAnotherVideoSection onTabChange={onTabChange} />
      </div>
      {!isMobile && (
        <MoreOptionsToolbarBottomActions onTabChange={onTabChange} />
      )}
    </div>
  );
}
