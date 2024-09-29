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
import LoadingStateScreen from '@/common/loading/LoadingStateScreen';
import DemoPage from '@/routes/DemoPage';
import stylex from '@stylexjs/stylex';
import {isFirefox} from 'react-device-detect';

const styles = stylex.create({
  link: {
    textDecorationLine: 'underline',
    color: '#A7B3BF',
  },
});

const REQUIRED_WINDOW_APIS = ['VideoEncoder', 'VideoDecoder', 'VideoFrame'];

function isBrowserSupported() {
  for (const api of REQUIRED_WINDOW_APIS) {
    if (!(api in window)) {
      return false;
    }
  }

  // Test if transferControlToOffscreen is supported. For example, this will
  // fail on iOS version < 16.4
  // https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/transferControlToOffscreen
  const canvas = document.createElement('canvas');
  if (typeof canvas.transferControlToOffscreen !== 'function') {
    return false;
  }

  return true;
}

export default function DemoPageWrapper() {
  const isBrowserUnsupported = !isBrowserSupported();

  if (isBrowserUnsupported && isFirefox) {
    const nightlyUrl = 'https://wiki.mozilla.org/Nightly';
    return (
      <LoadingStateScreen
        title="Sorry Firefox!"
        description={
          <div>
            This version of Firefox doesn’t support the video features we’ll
            need to run this demo. You can either update Firefox to the latest
            nightly build{' '}
            <a
              {...stylex.props(styles.link)}
              href={nightlyUrl}
              target="_blank"
              rel="noreferrer">
              here
            </a>
            , or try again using Chrome or Safari.
          </div>
        }
        linkProps={{to: '..', label: 'Back to homepage'}}
      />
    );
  }

  if (isBrowserUnsupported) {
    return (
      <LoadingStateScreen
        title="Uh oh, this browser isn’t supported."
        description="This browser doesn’t support the video features we’ll need to run this demo. Try again using Chrome, Safari, or Firefox Nightly."
        linkProps={{to: '..', label: 'Back to homepage'}}
      />
    );
  }

  return <DemoPage />;
}
