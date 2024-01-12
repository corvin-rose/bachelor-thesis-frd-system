import { TestBed } from '@angular/core/testing';

import { FrdService } from './frd.service';

describe('FrdService', () => {
  let service: FrdService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(FrdService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
