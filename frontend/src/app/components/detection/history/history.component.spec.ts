import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HistoryComponent } from './history.component';
import { HistoryService } from '../../../service/history.service';
import { Authenticity } from '../../../model/authenticity';
import { ClassificationResult } from '../../../model/classification-result';
import { By } from '@angular/platform-browser';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { MaterialModule } from '../../../material.module';
import { ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

describe('HistoryComponent', () => {
  let component: HistoryComponent;
  let fixture: ComponentFixture<HistoryComponent>;
  let historyServiceSpy: jasmine.SpyObj<HistoryService>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [HistoryComponent],
      imports: [
        HttpClientTestingModule,
        MaterialModule,
        ReactiveFormsModule,
        BrowserAnimationsModule,
      ],
      providers: [
        {
          provide: HistoryService,
          useValue: jasmine.createSpyObj('HistoryService', [
            'deleteFromHistory',
          ]),
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(HistoryComponent);
    component = fixture.componentInstance;
    historyServiceSpy = TestBed.inject(
      HistoryService,
    ) as jasmine.SpyObj<HistoryService>;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display history items correctly', () => {
    // given
    const mockHistory: ClassificationResult[] = [
      {
        input_text: 'Fake review 1',
        result: Authenticity.FAKE,
        probability: 0.98,
      },
      {
        input_text: 'Real review 2',
        result: Authenticity.REAL,
        probability: 0.96,
      },
    ];

    // when
    component.history = mockHistory;
    fixture.detectChanges();
    const historyItems = fixture.debugElement.queryAll(By.css('.history-item'));

    // then
    expect(historyItems.length).toEqual(mockHistory.length);
    historyItems.forEach((item, i) => {
      const itemText = item.query(By.css('span')).nativeElement.textContent;
      expect(itemText).toContain(component.trimText(mockHistory[i].input_text));
    });
  });

  it('should call deleteHistoryItem and emit historyUpdated on delete button click', () => {
    // given
    const mockHistoryItem: any = {
      key: 'history-key',
      input_text: 'Fake review 1',
      result: Authenticity.FAKE,
      probability: 0.98,
    };

    // when
    component.history = [mockHistoryItem];
    fixture.detectChanges();

    const deleteButton = fixture.debugElement.query(
      By.css(`button#${mockHistoryItem.key}`),
    ).nativeElement;
    deleteButton.click();

    // then
    fixture.whenStable().then(() => {
      expect(historyServiceSpy.deleteFromHistory).toHaveBeenCalledWith(
        mockHistoryItem.key,
      );
    });
    spyOn(component.historyUpdated, 'emit');

    // and when
    component.deleteHistoryItem(mockHistoryItem);

    // then
    expect(component.historyUpdated.emit).toHaveBeenCalled();
  });

  it('should call historyClicked on history item click', () => {
    // given
    const mockHistoryItem: any = {
      key: 'history-key',
      input_text: 'Fake review 1',
      result: Authenticity.FAKE,
      probability: 0.98,
    };
    component.history = [mockHistoryItem];
    fixture.detectChanges();

    // when
    spyOn(component.historyClicked, 'emit');
    const historyItem = fixture.debugElement.query(
      By.css('.history-item'),
    ).nativeElement;
    historyItem.click();

    // then
    expect(component.historyClicked.emit).toHaveBeenCalledWith(
      mockHistoryItem.key,
    );
  });

  it('should sort history items by timestamp', () => {
    // given
    const mockHistory: any[] = [
      {
        key: '1',
        classification: Authenticity.FAKE,
        input_text: 'Fake review 1',
        timestamp: '1706531067490',
      },
      {
        key: '2',
        classification: Authenticity.REAL,
        input_text: 'Real review 2',
        timestamp: '1706531067491',
      },
      {
        key: '3',
        classification: Authenticity.FAKE,
        input_text: 'Fake review 3',
        timestamp: '1706531067492',
      },
    ];
    component.history = mockHistory;
    fixture.detectChanges();

    // when
    const historyItems = component.historyList() as any[];

    // then
    expect(historyItems.length).toEqual(mockHistory.length);
    expect(historyItems[0].key).toEqual('3');
    expect(historyItems[1].key).toEqual('2');
    expect(historyItems[2].key).toEqual('1');
  });
});
