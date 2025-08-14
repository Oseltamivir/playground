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
/* ...existing code... */
export class AppendingLineChart {
  private numLines: number;
  private data: DataPoint[] = [];
  private svg;
  private xScale;
  private yScale;
  private paths;
  private smoothPaths;
  private lineColors: string[];
  private lightColors: string[];

  private minY = Number.MAX_VALUE;
  private maxY = Number.MIN_VALUE;

  private smoothingWindow: number = 20; 

  constructor(container, lineColors: string[]) {
    this.lineColors = lineColors;
    this.numLines = lineColors.length;
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
    this.minY = Number.MAX_VALUE;
    this.maxY = Number.MIN_VALUE;
  }

  addDataPoint(dataPoint: number[]) {
    if (dataPoint.length !== this.numLines) {
      throw Error("Length of dataPoint must equal number of lines");
    }
    dataPoint.forEach(y => {
      this.minY = Math.min(this.minY, y);
      this.maxY = Math.max(this.maxY, y);
    });

    this.data.push({x: this.data.length + 1, y: dataPoint});
    this.redraw();
  }

  private redraw() {
    // Adjust the x and y domain.
    this.xScale.domain([1, Math.max(2, this.data.length)]);
    this.yScale.domain([this.minY, this.maxY]);
    // Adjust all the <path> elements (lines).
    let getPathMap = (lineIndex: number) => {
      return d3.svg.line<{x: number, y:number[]}>()
        .x(d => this.xScale(d.x))
        .y(d => this.yScale(d.y[lineIndex]));
    };
    let getSmoothPathMap = (lineIndex: number) => {
      // Compute smoothed data for this line
      let smoothed = movingAverage(this.data.map(d => d.y[lineIndex]), this.smoothingWindow);
      let smoothData = this.data.map((d, i) => ({x: d.x, y: (() => {
        let arr = d.y.slice();
        arr[lineIndex] = smoothed[i];
        return arr;
      })()}));
      return d3.svg.line<{x: number, y:number[]}>()
        .x(d => this.xScale(d.x))
        .y(d => this.yScale(d.y[lineIndex]))(smoothData);
    };
    for (let i = 0; i < this.numLines; i++) {
      // Raw
      this.paths[i].datum(this.data).attr("d", getPathMap(i));
      // Smoothed
      this.smoothPaths[i].attr("d", getSmoothPathMap(i));
    }
  }
}
