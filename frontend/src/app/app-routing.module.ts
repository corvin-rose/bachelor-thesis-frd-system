import { RouterModule, Routes } from '@angular/router';
import { StartComponent } from './components/start/start.component';
import { DetectionComponent } from './components/detection/detection.component';
import { NgModule } from '@angular/core';

const routes: Routes = [
  { path: 'detection', component: DetectionComponent },
  { path: '**', component: StartComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
