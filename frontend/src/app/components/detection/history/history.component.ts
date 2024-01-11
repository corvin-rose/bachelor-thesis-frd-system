import { Component, Input } from '@angular/core';
import { NgForOf, NgIf } from '@angular/common';
import { MatListModule } from '@angular/material/list';
import { MaterialModule } from '../../../material.module';
import { MatRippleModule } from '@angular/material/core';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [NgIf, NgForOf, MatListModule, MaterialModule, MatRippleModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css',
})
export class HistoryComponent {
  @Input() history: string[] = [];
}
