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
import {InformationFilled} from '@carbon/icons-react';

export default function LimitNotice() {
  return (
    <div className="mt-6 gap-3 mx-6 flex items-center text-gray-400">
      <div>
        <InformationFilled size={32} />
      </div>
      <div className="text-sm leading-snug">
        In this demo, you can track up to 3 objects, even though the SAM 2 model
        does not have a limit.
      </div>
    </div>
  );
}
