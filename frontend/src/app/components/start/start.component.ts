import { Component } from '@angular/core';
import { MaterialModule } from '../../material.module';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-start',
  standalone: true,
  imports: [MaterialModule, RouterLink],
  templateUrl: './start.component.html',
  styleUrl: './start.component.css',
})
export class StartComponent {}
