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
import useReportError from '@/common/error/useReportError';
import {Button} from 'react-daisyui';
import {FallbackProps} from 'react-error-boundary';

export default function ErrorFallback({
  error,
  resetErrorBoundary,
}: FallbackProps) {
  const reportError = useReportError();

  function handleReportError() {
    reportError(error);
  }

  return (
    <div className="h-full flex flex-col gap-2 items-center justify-center">
      <p>Please check your connection and retry or report error.</p>
      <div className="flex flex-row gap-2">
        <Button color="ghost" onClick={resetErrorBoundary}>
          Retry
        </Button>
        <Button
          className="text-error"
          color="ghost"
          onClick={handleReportError}>
          Report Error
        </Button>
      </div>
    </div>
  );
}
