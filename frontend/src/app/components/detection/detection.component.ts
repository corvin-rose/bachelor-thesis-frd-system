import { Component } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { RouterLink } from '@angular/router';
import { MaterialModule } from '../../material.module';
import { HistoryComponent } from './history/history.component';

@Component({
  selector: 'app-detection',
  standalone: true,
  imports: [MatButtonModule, RouterLink, MaterialModule, HistoryComponent],
  templateUrl: './detection.component.html',
  styleUrl: './detection.component.css',
})
export class DetectionComponent {
  history: string[] = ['test 1', 'test 2', 'test 3'];
}
