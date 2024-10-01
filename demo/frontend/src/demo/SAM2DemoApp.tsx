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
import '@/assets/scss/App.scss';
import ErrorReport from '@/common/error/ErrorReport';
import DemoErrorFallback from '@/demo/DemoErrorFallback';
import DemoSuspenseFallback from '@/demo/DemoSuspenseFallback';
import RelayEnvironmentProvider from '@/graphql/RelayEnvironmentProvider';
import RootLayout from '@/layouts/RootLayout';
import SAM2DemoPage from '@/routes/DemoPageWrapper';
import PageNotFoundPage from '@/routes/PageNotFoundPage';
import useSettingsContext from '@/settings/useSettingsContext';
import {Route, Routes} from 'react-router-dom';

export default function DemoAppWrapper() {
  const {settings} = useSettingsContext();
  return (
    <RelayEnvironmentProvider
      endpoint={settings.videoAPIEndpoint}
      suspenseFallback={<DemoSuspenseFallback />}
      errorFallback={DemoErrorFallback}>
      <DemoApp />
    </RelayEnvironmentProvider>
  );
}

function DemoApp() {
  return (
    <>
      <Routes>
        <Route element={<RootLayout />}>
          <Route index={true} element={<SAM2DemoPage />} />
          <Route path="*" element={<PageNotFoundPage />} />
        </Route>
      </Routes>
      <ErrorReport />
    </>
  );
}
