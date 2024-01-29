import { TestBed } from '@angular/core/testing';

import { HISTORY_STORAGE_KEY, HistoryService } from './history.service';

describe('HistoryService', () => {
  let service: HistoryService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(HistoryService);
    localStorage.clear();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should save data to localStorage and return saved data with generated key', () => {
    // given
    const testData = { v: 'Test' };

    // when
    const saved = service.saveToHistory(testData);
    delete saved.timestamp;

    // then
    const keysInLocalStorage = Object.keys(localStorage);
    expect(keysInLocalStorage.length).toBe(1);
    expect(keysInLocalStorage[0]).toContain(HISTORY_STORAGE_KEY);

    const loaded = service.loadFromHistory()[0];
    delete loaded.timestamp;
    expect(loaded).toEqual({ key: keysInLocalStorage[0], ...testData });
    expect(saved).toEqual({ key: keysInLocalStorage[0], ...testData });
  });

  it('should delete data from localStorage by key', () => {
    // given
    const testData = { v: 'Test' };
    const saved = service.saveToHistory(testData);
    const key = saved.key;

    // when
    service.deleteFromHistory(key);

    // then
    expect(localStorage.getItem(key)).toBeNull();
  });

  it('should load data from localStorage', () => {
    // given
    const testData1 = { v: 'Test 1' };
    const testData2 = { v: 'Test 2' };
    const saved1 = service.saveToHistory(testData1);
    const saved2 = service.saveToHistory(testData2);

    // when
    const loaded = service.loadFromHistory();

    // then
    expect(loaded.length).toBe(2);
    expect(loaded).toContain(saved1);
    expect(loaded).toContain(saved2);
  });

  it('should generate unique random ID', () => {
    // when
    const id = service.generateId();

    // then
    expect(id.length).toBe(8);
    expect(id).toMatch(/^[0-9a-f]+$/);
    expect(id).not.toEqual(service.generateId());
  });
});
