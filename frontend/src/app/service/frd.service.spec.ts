import { TestBed } from '@angular/core/testing';

import { FrdService } from './frd.service';
import { RouterTestingModule } from '@angular/router/testing';
import {
  HttpClientTestingModule,
  HttpTestingController,
} from '@angular/common/http/testing';
import { ClassificationResult } from '../model/classification-result';
import { environment } from '../app.config';
import { Authenticity } from '../model/authenticity';

describe('FrdService', () => {
  let service: FrdService;
  let httpTestingController: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
    });
    service = TestBed.inject(FrdService);
    httpTestingController = TestBed.inject(HttpTestingController);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should send a POST request to correct URL with given review', () => {
    // given
    const testReview = 'Test review';
    const mockResult: ClassificationResult = {
      input_text: testReview,
      result: Authenticity.FAKE,
      probability: 0.98,
    };

    // when
    service.checkReview(testReview).subscribe((result) => {
      expect(result).toEqual(mockResult);
    });

    // then
    const req = httpTestingController.expectOne(
      `${environment.apiBaseUrl}/frd/`,
    );
    expect(req.request.method).toEqual('POST');
    expect(req.request.body).toEqual(JSON.stringify(testReview));

    req.flush(mockResult);

    httpTestingController.verify();
  });
});
