/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * Main application component
 */

import { ViewerLayout } from './components/viewer/ViewerLayout';
import { UpgradePage } from './components/viewer/UpgradePage';
import { StreamlitBridge } from './components/streamlit/StreamlitBridge';
import { BimProvider } from './sdk/BimProvider';
import { Toaster } from './components/ui/toast';
import { ClerkChatSync } from './lib/llm/ClerkChatSync';
import { isClerkConfigured } from './lib/llm/clerk-auth';
import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from 'react';

interface ViewerErrorBoundaryState {
  hasError: boolean;
}

class ViewerErrorBoundary extends Component<{ children: ReactNode }, ViewerErrorBoundaryState> {
  state: ViewerErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ViewerErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[Viewer] Unhandled rendering error', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen w-screen items-center justify-center bg-background p-6 text-foreground">
          <div className="max-w-md text-center">
            <h1 className="text-xl font-semibold">Viewer crashed</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              An unexpected error occurred while rendering the 3D viewer.
            </p>
            <button
              type="button"
              className="mt-4 rounded-md bg-primary px-4 py-2 text-primary-foreground"
              onClick={() => window.location.reload()}
            >
              Reload page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export function App() {
  const clerkEnabled = isClerkConfigured();
  const [pathname, setPathname] = useState(() => window.location.pathname);

  useEffect(() => {
    const onRouteChange = () => setPathname(window.location.pathname);
    window.addEventListener('popstate', onRouteChange);
    return () => window.removeEventListener('popstate', onRouteChange);
  }, []);

  const isUpgradeRoute = pathname === '/upgrade';

  return (
    <BimProvider>
      <StreamlitBridge />
      {clerkEnabled && <ClerkChatSync />}
      {isUpgradeRoute ? (
        <UpgradePage />
      ) : (
        <ViewerErrorBoundary>
          <ViewerLayout />
        </ViewerErrorBoundary>
      )}
      <Toaster />
    </BimProvider>
  );
}

export default App;
