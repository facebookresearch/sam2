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
import {EnableStatsRequest} from '@/common/components/video/VideoWorkerTypes';
import stylex from '@stylexjs/stylex';
import {useEffect, useMemo, useRef, useState} from 'react';
import {useLocation} from 'react-router-dom';
import useVideo from '../../common/components/video/editor/useVideo';
import {
  GetMemoryStatsRequest,
  GetStatsCanvasRequest,
  MemoryStatsResponse,
  SetStatsCanvasResponse,
} from './Stats';

const styles = stylex.create({
  container: {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    overflowX: 'auto',
    display: 'flex',
    flexDirection: 'row',
    cursor: 'pointer',
    opacity: 0.9,
    zIndex: 10000,
  },
});

const URL_PARAM = 'monitors';

export default function StatsView() {
  const {search} = useLocation();
  const video = useVideo();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isWrapped, setIsWrapped] = useState<boolean>(false);

  const isEnabled = useMemo(() => {
    const urlSearchParams = new URLSearchParams(search);
    return (
      urlSearchParams.has(URL_PARAM) &&
      ['true', ''].includes(urlSearchParams.get('monitors') ?? '')
    );
  }, [search]);

  useEffect(() => {
    if (!isEnabled) {
      return;
    }

    const worker = video?.getWorker_ONLY_USE_WITH_CAUTION();

    // Enable stats for video worker
    worker?.postMessage({
      action: 'enableStats',
    } as EnableStatsRequest);

    function onMessage(
      event: MessageEvent<GetStatsCanvasRequest | GetMemoryStatsRequest>,
    ) {
      if (event.data.action === 'getStatsCanvas') {
        // Add stats canvas and hand control over to worker
        const canvas = document.createElement('canvas');
        canvas.width = event.data.width * window.devicePixelRatio;
        canvas.height = event.data.height * window.devicePixelRatio;
        canvas.style.width = `${event.data.width}px`;
        canvas.style.height = `${event.data.height}px`;
        containerRef.current?.appendChild(canvas);

        const offscreenCanvas = canvas.transferControlToOffscreen();
        worker?.postMessage(
          {
            action: 'setStatsCanvas',
            id: event.data.id,
            canvas: offscreenCanvas,
            devicePixelRatio: window.devicePixelRatio,
          } as SetStatsCanvasResponse,
          {
            transfer: [offscreenCanvas],
          },
        );
      } else if (event.data.action === 'getMemoryStats') {
        // @ts-expect-error performance.memory might not exist
        const memory = performance.memory ?? {
          jsHeapSizeLimit: 0,
          totalJSHeapSize: 0,
          usedJSHeapSize: 0,
        };
        worker?.postMessage({
          action: 'memoryStats',
          id: event.data.id,
          jsHeapSizeLimit: memory.jsHeapSizeLimit,
          totalJSHeapSize: memory.totalJSHeapSize,
          usedJSHeapSize: memory.usedJSHeapSize,
        } as MemoryStatsResponse);
      }
    }

    worker?.addEventListener('message', onMessage);
    return () => {
      worker?.removeEventListener('message', onMessage);
    };
  }, [video, isEnabled]);

  function handleClick() {
    setIsWrapped(w => !w);
  }

  if (!isEnabled) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      {...stylex.props(styles.container)}
      style={{flexWrap: isWrapped ? 'wrap' : 'unset'}}
      onDoubleClick={handleClick}
    />
  );
}
