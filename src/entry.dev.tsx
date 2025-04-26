/*
 * WHAT IS THIS FILE?
 *
 * Development entry point using only client-side modules:
 * - Do not use this in production
 * - No SSR
 * - No portion of this should be included in the production build
 */

import { render, type RenderOptions } from '@builder.io/qwik';
import Root from './root';

export default function (opts: RenderOptions) {
  return render(document, <Root />, opts);
}
