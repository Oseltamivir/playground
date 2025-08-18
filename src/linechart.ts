/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as d3 from 'd3';

type DataPoint = {
  x: number;
  y: number[];
};

interface LineChartOptions {
  secondarySeries?: number[];        // indices rendered on right Y axis
  y2Domain?: [number, number];       // fixed domain for right axis (e.g., [0,1])
  smoothingWindow?: number;          // override default smoothing window
}

function movingAverage(data: number[], window: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < data.length; i++) {
    let start = Math.max(0, i - window + 1);
    let sum = 0;
    for (let j = start; j <= i; j++) sum += data[j];
    result.push(sum / (i - start + 1));
  }
  return result;
}

/**
 * A multi-series line chart that allows you to append new data points
 * as data becomes available.
 */
export class AppendingLineChart {
  private numLines: number;
  private data: DataPoint[] = [];
  private svg;
  private xScale;
  private yScale;
  private y2Scale;                   // NEW: right-axis scale
  private paths;
  private smoothPaths;
  private lineColors: string[];
  private lightColors: string[];

  // Track domains separately for primary (losses) and secondary (accuracy)
  private minY1 = Number.POSITIVE_INFINITY;
  private maxY1 = Number.NEGATIVE_INFINITY;
  private minY2 = Number.POSITIVE_INFINITY;
  private maxY2 = Number.NEGATIVE_INFINITY;

  private smoothingWindow: number = 10;

  private secondarySet: Set<number> = new Set(); // which series use y2
  private y2Fixed?: [number, number];            // fixed domain for y2 if provided

  constructor(container, lineColors: string[], opts: LineChartOptions = {}) {
    this.lineColors = lineColors;
    this.numLines = lineColors.length;
    this.secondarySet = new Set(opts.secondarySeries || []);
    this.y2Fixed = opts.y2Domain;
    if (typeof opts.smoothingWindow === "number") {
      this.smoothingWindow = opts.smoothingWindow!;
    }
    // Generate lighter colors for background
    this.lightColors = lineColors.map(c => d3.rgb(c).brighter(2).toString());

    let node = container.node() as HTMLElement;
    let totalWidth = node.offsetWidth;
    let totalHeight = node.offsetHeight;
    let margin = {top: 2, right: 0, bottom: 2, left: 2};
    let width = totalWidth - margin.left - margin.right;
    let height = totalHeight - margin.top - margin.bottom;

    this.xScale = d3.scale.linear()
      .domain([0, 0])
      .range([0, width]);

    this.yScale = d3.scale.linear()
      .domain([0, 0])
      .range([height, 0]);

    // NEW: right axis scale (accuracy)
    this.y2Scale = d3.scale.linear()
      .domain(this.y2Fixed || [0, 0])
      .range([height, 0]);

    this.svg = container.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Background (raw) lines
    this.paths = new Array(this.numLines);
    // Foreground (smoothed) lines
    this.smoothPaths = new Array(this.numLines);

    for (let i = 0; i < this.numLines; i++) {
      // Raw data (background)
      this.paths[i] = this.svg.append("path")
        .attr("class", "line raw")
        .style({
          "fill": "none",
          "stroke": this.lightColors[i],
          "stroke-width": "1.5px",
          "stroke-opacity": 0.4
        });
      // Smoothed data (foreground)
      this.smoothPaths[i] = this.svg.append("path")
        .attr("class", "line smooth")
        .style({
          "fill": "none",
          "stroke": this.lineColors[i],
          "stroke-width": "1.5px"
        });
    }
  }

  getData(): DataPoint[] {
    return this.data;
  }

  setSmoothingWindow(window: number) {
    this.smoothingWindow = window;
    this.redraw();
  }

  reset() {
    this.data = [];
    this.redraw();
    this.minY1 = Number.POSITIVE_INFINITY;
    this.maxY1 = Number.NEGATIVE_INFINITY;
    this.minY2 = Number.POSITIVE_INFINITY;
    this.maxY2 = Number.NEGATIVE_INFINITY;
  }

  addDataPoint(dataPoint: number[]) {
    if (dataPoint.length !== this.numLines) {
      throw Error("Length of dataPoint must equal number of lines");
    }

    // Update running domains separately for primary vs secondary series
    dataPoint.forEach((y, i) => {
      if (this.secondarySet.has(i)) {
        this.minY2 = Math.min(this.minY2, y);
        this.maxY2 = Math.max(this.maxY2, y);
      } else {
        this.minY1 = Math.min(this.minY1, y);
        this.maxY1 = Math.max(this.maxY1, y);
      }
    });

    this.data.push({x: this.data.length + 1, y: dataPoint});
    this.redraw();
  }

  private redraw() {
    // Adjust the x domain.
    this.xScale.domain([1, Math.max(2, this.data.length)]);

    // Primary Y: losses (relative scale)
    const hasPrimary = isFinite(this.minY1) && isFinite(this.maxY1);
    this.yScale.domain(hasPrimary ? [this.minY1, this.maxY1] : [0, 1]);

    // Secondary Y: accuracy (fixed [0,1] if provided, else relative)
    if (this.y2Fixed) {
      this.y2Scale.domain(this.y2Fixed);
    } else {
      const hasSecondary = isFinite(this.minY2) && isFinite(this.maxY2);
      this.y2Scale.domain(hasSecondary ? [this.minY2, this.maxY2] : [0, 1]);
    }

    // Helpers to map line series to the appropriate Y scale
    const pathMap = (lineIndex: number) => {
      const scale = this.secondarySet.has(lineIndex) ? this.y2Scale : this.yScale;
      return d3.svg.line<{x: number, y:number[]}>()
        .x(d => this.xScale(d.x))
        .y(d => scale(d.y[lineIndex]));
    };

    const smoothPathD = (lineIndex: number) => {
      const scale = this.secondarySet.has(lineIndex) ? this.y2Scale : this.yScale;
      // Compute smoothed data for this line
      let smoothed = movingAverage(this.data.map(d => d.y[lineIndex]), this.smoothingWindow);
      let smoothData = this.data.map((d, i) => {
        let arr = d.y.slice();
        arr[lineIndex] = smoothed[i];
        return { x: d.x, y: arr };
      });
      return d3.svg.line<{x: number, y:number[]}>()
        .x(d => this.xScale(d.x))
        .y(d => scale(d.y[lineIndex]))(smoothData);
    };

    for (let i = 0; i < this.numLines; i++) {
      // Raw
      this.paths[i].datum(this.data).attr("d", pathMap(i));
      // Smoothed
      this.smoothPaths[i].attr("d", smoothPathD(i));
    }
  }
}