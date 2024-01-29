import { Injectable } from '@angular/core';

export const HISTORY_STORAGE_KEY = 'history-storage';

@Injectable({
  providedIn: 'root',
})
export class HistoryService {
  constructor() {}

  saveToHistory(data: any) {
    const key = this.generateStorageKey();
    const dataToSave = { timestamp: Date.now(), ...data };
    localStorage.setItem(key, JSON.stringify(dataToSave));
    return { key: key, ...dataToSave };
  }

  deleteFromHistory(key: string): void {
    localStorage.removeItem(key);
  }

  loadFromHistory(): any {
    return Array(localStorage.length)
      .fill('')
      .map((_, i) => localStorage.key(i))
      .filter((key) => key?.includes(HISTORY_STORAGE_KEY))
      .map((key) => [key, localStorage.getItem(key ?? '')])
      .map(([key, data]) => {
        return { key: key, ...(data ? JSON.parse(data) : null) };
      });
  }

  generateStorageKey(): string {
    return `${HISTORY_STORAGE_KEY}-${this.generateId()}`;
  }

  generateId(): string {
    return Array(8)
      .fill(0)
      .map((_) => Math.floor(Math.random() * 16).toString(16))
      .join('');
  }
}
