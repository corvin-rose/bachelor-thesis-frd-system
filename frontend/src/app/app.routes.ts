import { Routes } from '@angular/router';
import { StartComponent } from './components/start/start.component';
import { DetectionComponent } from './components/detection/detection.component';

export const routes: Routes = [
  { path: 'detection', component: DetectionComponent },
  { path: '**', component: StartComponent },
];
