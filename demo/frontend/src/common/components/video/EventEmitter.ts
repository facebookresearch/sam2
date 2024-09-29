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
type EventMap<WorkerEventMap> = {
  type: keyof WorkerEventMap;
  listener: (ev: WorkerEventMap[keyof WorkerEventMap]) => unknown;
};

export class EventEmitter<WorkerEventMap> {
  listeners: EventMap<WorkerEventMap>[] = [];

  trigger<K extends keyof WorkerEventMap>(type: K, ev: WorkerEventMap[K]) {
    this.listeners
      .filter(listener => type === listener.type)
      .forEach(({listener}) => {
        setTimeout(() => listener(ev), 0);
      });
  }

  addEventListener<K extends keyof WorkerEventMap>(
    type: K,
    listener: (ev: WorkerEventMap[K]) => unknown,
  ): void {
    // @ts-expect-error Incorrect typing. Not sure how to correctly type it
    this.listeners.push({type, listener});
  }

  removeEventListener<K extends keyof WorkerEventMap>(
    type: K,
    listener: (ev: WorkerEventMap[K]) => unknown,
  ): void {
    this.listeners = this.listeners.filter(
      existingListener =>
        !(
          existingListener.type === type &&
          existingListener.listener === listener
        ),
    );
  }

  destroy() {
    this.listeners.length = 0;
  }
}
