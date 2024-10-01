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
import {CanvasForm, CanvasSpace, Font, Group, Pt, Triangle} from 'pts';
import SelectedFrameHelper from './SelectedFrameHelper';
import {PADDING_BOTTOM, PADDING_TOP} from './VideoFilmstrip';

export function getPointerPosition(
  event: React.PointerEvent<HTMLCanvasElement>,
) {
  const rect = event.currentTarget.getBoundingClientRect();
  return new Pt(event.clientX - rect.left, event.clientY - rect.top);
}

export function drawFilmstrip(
  filmstrip: ImageBitmap | null,
  space: CanvasSpace | undefined,
  form: CanvasForm | undefined,
) {
  if (filmstrip == null || space == undefined || form?.ctx == undefined) {
    return;
  }

  const ratio =
    filmstrip.width / (filmstrip.height + PADDING_TOP + PADDING_BOTTOM);

  form.image(
    [
      [0, PADDING_TOP],
      [space.size.x, space.size.x / ratio],
    ],
    filmstrip,
  );
}

export function getTimeFromFrame(frame: number, fps: number): string {
  const seconds = Math.floor(frame / fps);
  const frameRemaining = frame - fps * seconds;
  return `${seconds}:${frameRemaining.toFixed().toString().padStart(2, '0')}`;
}

export function drawMarker(
  space: CanvasSpace | undefined,
  form: CanvasForm | undefined,
  selectedFrameHelper: SelectedFrameHelper,
  pointerPosition: Pt | null,
  scanLabel: string | false,
  fps: number,
) {
  if (space == undefined || form?.ctx == undefined) {
    return;
  }

  const marker = Group.fromArray([
    [0, PADDING_TOP],
    [0, space.height - PADDING_BOTTOM],
  ]);

  const currentMarker = marker
    .clone()
    .add(Math.max(5, selectedFrameHelper.position), 0);

  const getTextPosition = (label: string, marker: Group) => {
    const textWidth = form.ctx.measureText(label).width;
    return marker[0]
      .$subtract(textWidth / 2, 0)
      .$min(space.width - textWidth, PADDING_TOP - 10)
      .$max(textWidth / 2 - 2, 0);
  };

  // draw current marker
  form
    .strokeOnly('#00000066', 5)
    .line(currentMarker)
    .strokeOnly('#fff', 1)
    .line(currentMarker)
    .fill('#000')
    .polygon(
      Triangle.fromCenter(currentMarker[0].$add(0, 10), 5).rotate2D(Math.PI),
    );

  // draw text
  const frameLabel = getTimeFromFrame(selectedFrameHelper.index, fps);
  form
    .font(new Font(12, 'monospace'))
    .fillOnly('#fff')
    .text(getTextPosition(frameLabel, currentMarker), frameLabel);

  // draw scanning ghost marker
  if (
    selectedFrameHelper.isScanning &&
    pointerPosition != null &&
    scanLabel != false
  ) {
    const scanMarker = marker.clone().add(pointerPosition.x, 0);
    form.strokeOnly('#ffffff66', 5).line(scanMarker);

    form
      .font(new Font(12, 'monospace'))
      .fillOnly('#8595A4')
      .text(getTextPosition(scanLabel, scanMarker), scanLabel);
  }
}
