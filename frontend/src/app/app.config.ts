import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';

import { provideAnimations } from '@angular/platform-browser/animations';

export const appConfig: ApplicationConfig = {
  providers: [provideRouter([]), provideAnimations()],
};

export const environment = {
  apiBaseUrl: 'http://127.0.0.1:8000',
};
