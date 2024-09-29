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
import {useCallback, useState} from 'react';

type ThrottleOptions = {
  enableThrottling?: boolean;
};
type State = {
  isThrottled: boolean;
  maxThrottles: boolean;
  throttle: (callback: () => void, options?: ThrottleOptions) => void;
};

export default function useFunctionThrottle(
  initialDelay: number,
  numThrottles: number,
): State {
  const [isThrottled, setIsThrottled] = useState<boolean>(false);
  const [lastClickTime, setLastClickTime] = useState<number | null>(null);
  const [numTimesThrottled, setNumTimesThrottled] = useState<number>(1);

  /**
   * The following function's callback gets throttled when the time between two
   * executions is less than a threshold.
   *
   * The threshold is calculated linearly by multiplying the initial delay
   * and the number of times the button has been throttled. The button can be
   * throttled up to numThrottles times.
   *
   * The function has an optional flag - enableThrottling - which allows a callsite
   * to optionally disable throttling. This is useful in cases where throttling may
   * not be necessary. (e.g. for the Track & Play button, we would only like to
   * throttle after a stream is aborted.)
   */
  const throttle = useCallback(
    (
      callback: () => void,
      options: ThrottleOptions = {
        enableThrottling: true,
      },
    ) => {
      if (isThrottled) {
        return;
      }

      const currentTime = Date.now();
      if (lastClickTime == null) {
        callback();
        setLastClickTime(currentTime);
        return;
      }

      const timeBetweenClicks = currentTime - lastClickTime;
      const delay = initialDelay * numTimesThrottled;
      const shouldThrottle =
        options.enableThrottling && delay > timeBetweenClicks;

      if (shouldThrottle) {
        setIsThrottled(true);
        setTimeout(() => {
          setIsThrottled(false);
        }, delay);
        setNumTimesThrottled(prev => {
          return prev === numThrottles ? numThrottles : prev + 1;
        });
      }

      callback();
      setLastClickTime(currentTime);
    },
    [initialDelay, numThrottles, isThrottled, lastClickTime, numTimesThrottled],
  );

  return {
    isThrottled,
    maxThrottles: numTimesThrottled === numThrottles,
    throttle,
  };
}
