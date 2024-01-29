import { Component, EventEmitter, Input, Output } from '@angular/core';
import { ClassificationResult } from '../../../model/classification-result';
import { HistoryService } from '../../../service/history.service';
import { Authenticity } from '../../../model/authenticity';

@Component({
  selector: 'app-history',
  templateUrl: './history.component.html',
  styleUrl: './history.component.css',
})
export class HistoryComponent {
  @Input() history: ClassificationResult[] = [];
  @Output() historyClicked = new EventEmitter<string>();
  @Output() historyUpdated = new EventEmitter<void>();

  constructor(private historyService: HistoryService) {}

  deleteHistoryItem(item: any): void {
    this.historyService.deleteFromHistory(item.key);
    this.historyUpdated.emit();
  }

  itemClick(event: any, item: any): void {
    if (
      [...event.target.classList].filter((c: string) => c.includes('button'))
        .length == 0
    ) {
      this.historyClicked.emit(item.key);
    }
  }

  trimText(text: string): string {
    return text.length >= 30 ? text.substring(0, 30) + '...' : text;
  }

  historyList(): any[] {
    return (this.history ?? [])
      .map((h) => h as any)
      .sort((a, b) => +b.timestamp - +a.timestamp);
  }

  protected readonly Authenticity = Authenticity;
}
