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
import ErrorFallback from '@/common/error/ErrorFallback';
import LoadingMessage from '@/common/loading/LoadingMessage';
import {createEnvironment} from '@/graphql/RelayEnvironment';
import {
  ComponentType,
  PropsWithChildren,
  ReactNode,
  Suspense,
  useMemo,
  useState,
} from 'react';
import {ErrorBoundary, FallbackProps} from 'react-error-boundary';
import {RelayEnvironmentProvider} from 'react-relay';

type Props = PropsWithChildren<{
  suspenseFallback?: ReactNode;
  errorFallback?: ComponentType<FallbackProps>;
  endpoint: string;
}>;

export default function OnevisionRelayEnvironmentProvider({
  suspenseFallback,
  errorFallback = ErrorFallback,
  endpoint,
  children,
}: Props) {
  const [retryKey, setRetryKey] = useState<number>(0);

  const environment = useMemo(() => {
    return createEnvironment(endpoint);
    // The retryKey is needed to force a new Relay Environment
    // instance when the user retries after an error occurred.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, retryKey]);

  // Force re-creating Relay Environment
  function handleReset() {
    setRetryKey(k => k + 1);
  }

  return (
    <ErrorBoundary onReset={handleReset} FallbackComponent={errorFallback}>
      <RelayEnvironmentProvider environment={environment}>
        <Suspense fallback={suspenseFallback ?? <LoadingMessage />}>
          {children}
        </Suspense>
      </RelayEnvironmentProvider>
    </ErrorBoundary>
  );
}
