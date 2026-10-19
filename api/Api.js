{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "requestAttributes": {
      "time": "2026-10-19T18:43:02.490517Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "clientauthconfig.googleapis.com",
  "methodName": "CreateBrand",
  "authorizationInfo": [{
    "resource": "brands/992786648543",
    "permission": "clientauthconfig.brands.create",
    "granted": true,
    "resourceAttributes": {
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "brands/992786648543",
  "request": {
    "@type": "type.googleapis.com/google.identity.clientauthconfig.v1.CreateBrandRequest",
    "brand": {
      "supportEmail": "chosen06021957@gmail.com"
    }
  },
  "response": {
    "@type": "type.googleapis.com/google.identity.clientauthconfig.v1.Brand",
    "brandId": "992786648543",
    "supportEmail": "chosen06021957@gmail.com"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:48:05.133462Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.CreateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.create",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543",
      "type": "cloudresourcemanager.googleapis.com/Project"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763",
  "request": {
    "parent": "projects/gen-lang-client-0049300763",
    "dashboard": {
      "displayName": "Ai",
      "mosaicLayout": {
        "tiles": [{
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "dataSets": [{
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM",
                      "groupByFields": ["resource.labels.method"]
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                },
                "plotType": "LINE"
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "width": 6.0,
          "height": 4.0
        }],
        "columns": 12.0
      }
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.CreateDashboardRequest"
  },
  "response": {
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "mosaicLayout": {
      "tiles": [{
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "dataSets": [{
              "plotType": "LINE",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM",
                    "groupByFields": ["resource.labels.method"]
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              }
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "height": 4.0,
        "width": 6.0
      }],
      "columns": 12.0
    },
    "displayName": "Ai",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:48:23.670035Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "dashboard": {
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "mosaicLayout": {
        "tiles": [{
          "width": 6.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "groupByFields": ["resource.labels.method"]
                    }
                  }
                }
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "height": 4.0
        }, {
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p50"
              }, {
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                      "perSeriesAligner": "ALIGN_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                },
                "legendTemplate": "p95"
              }, {
                "legendTemplate": "p99",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE"
              }]
            }
          },
          "height": 4.0,
          "width": 6.0,
          "xPos": 6.0
        }],
        "columns": 12.0
      },
      "displayName": "Ai"
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest"
  },
  "response": {
    "mosaicLayout": {
      "tiles": [{
        "width": 6.0,
        "height": 4.0,
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM",
                    "groupByFields": ["resource.labels.method"]
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "targetAxis": "Y1"
            }]
          }
        }
      }, {
        "xPos": 6.0,
        "width": 6.0,
        "widget": {
          "xyChart": {
            "dataSets": [{
              "legendTemplate": "p50",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "plotType": "LINE"
            }, {
              "legendTemplate": "p95",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              },
              "targetAxis": "Y1"
            }, {
              "targetAxis": "Y1",
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              },
              "plotType": "LINE"
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          },
          "title": "Overall latency"
        },
        "height": 4.0
      }],
      "columns": 12.0
    },
    "displayName": "Ai",
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:48:53.449760Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "dashboard": {
      "displayName": "Ai",
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "mosaicLayout": {
        "columns": 12.0,
        "tiles": [{
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "groupByFields": ["resource.labels.method"],
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1"
              }]
            }
          },
          "width": 6.0,
          "height": 4.0
        }, {
          "xPos": 6.0,
          "widget": {
            "xyChart": {
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "legendTemplate": "p50",
                "plotType": "LINE"
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "plotType": "LINE",
                "legendTemplate": "p95"
              }, {
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p99",
                "plotType": "LINE"
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Overall latency"
          },
          "width": 6.0,
          "height": 4.0
        }, {
          "width": 6.0,
          "yPos": 4.0,
          "height": 4.0,
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "groupByFields": ["resource.labels.method"],
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE"
              }]
            },
            "title": "Latency by API method (median)"
          }
        }, {
          "yPos": 4.0,
          "height": 4.0,
          "width": 6.0,
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "legendTemplate": "p50",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE"
              }, {
                "plotType": "LINE",
                "legendTemplate": "p95",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                }
              }, {
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  }
                },
                "legendTemplate": "p99"
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "xPos": 6.0
        }]
      }
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest"
  },
  "response": {
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
    "mosaicLayout": {
      "columns": 12.0,
      "tiles": [{
        "width": 6.0,
        "height": 4.0,
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "groupByFields": ["resource.labels.method"]
                  }
                },
                "unitOverride": "second"
              }
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        }
      }, {
        "xPos": 6.0,
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1",
              "legendTemplate": "p50",
              "plotType": "LINE"
            }, {
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "plotType": "LINE",
              "legendTemplate": "p95",
              "targetAxis": "Y1"
            }, {
              "plotType": "LINE",
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1"
            }]
          }
        },
        "height": 4.0,
        "width": 6.0
      }, {
        "height": 4.0,
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "groupByFields": ["resource.labels.method"]
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "yPos": 4.0,
        "width": 6.0
      }, {
        "yPos": 4.0,
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "legendTemplate": "p50",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              }
            }, {
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                  }
                }
              },
              "legendTemplate": "p95"
            }, {
              "legendTemplate": "p99",
              "plotType": "LINE",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  }
                }
              }
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "height": 4.0,
        "width": 6.0,
        "xPos": 6.0
      }]
    },
    "displayName": "Ai"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:49:04.410310Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "dashboard": {
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "mosaicLayout": {
        "tiles": [{
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "groupByFields": ["resource.labels.method"]
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                }
              }]
            },
            "title": "Latency by API method (median)"
          },
          "width": 6.0,
          "height": 4.0
        }, {
          "width": 6.0,
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE",
                "legendTemplate": "p50"
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "legendTemplate": "p95",
                "plotType": "LINE"
              }, {
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  }
                },
                "legendTemplate": "p99"
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "height": 4.0,
          "xPos": 6.0
        }, {
          "width": 6.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "groupByFields": ["resource.labels.method"],
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                }
              }]
            }
          },
          "height": 4.0,
          "yPos": 4.0
        }, {
          "xPos": 6.0,
          "width": 6.0,
          "height": 4.0,
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "legendTemplate": "p50",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }, {
                "legendTemplate": "p95",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }, {
                "targetAxis": "Y1",
                "plotType": "LINE",
                "legendTemplate": "p99",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    }
                  },
                  "unitOverride": "second"
                }
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Overall latency"
          },
          "yPos": 4.0
        }, {
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    },
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    }
                  }
                }
              }]
            },
            "title": "Errors by API method"
          },
          "yPos": 8.0,
          "width": 6.0,
          "height": 4.0
        }],
        "columns": 12.0
      },
      "displayName": "Ai"
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest"
  },
  "response": {
    "mosaicLayout": {
      "columns": 12.0,
      "tiles": [{
        "widget": {
          "xyChart": {
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "groupByFields": ["resource.labels.method"],
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1"
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          },
          "title": "Latency by API method (median)"
        },
        "width": 6.0,
        "height": 4.0
      }, {
        "xPos": 6.0,
        "height": 4.0,
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                }
              },
              "legendTemplate": "p50",
              "targetAxis": "Y1",
              "plotType": "LINE"
            }, {
              "legendTemplate": "p95",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              },
              "plotType": "LINE"
            }, {
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  }
                },
                "unitOverride": "second"
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "width": 6.0
      }, {
        "height": 4.0,
        "width": 6.0,
        "widget": {
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "groupByFields": ["resource.labels.method"],
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                }
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }]
          },
          "title": "Latency by API method (median)"
        },
        "yPos": 4.0
      }, {
        "xPos": 6.0,
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "targetAxis": "Y1",
              "legendTemplate": "p50",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              }
            }, {
              "legendTemplate": "p95",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1"
            }, {
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }]
          }
        },
        "width": 6.0,
        "height": 4.0,
        "yPos": 4.0
      }, {
        "yPos": 8.0,
        "width": 6.0,
        "widget": {
          "title": "Errors by API method",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  }
                }
              },
              "targetAxis": "Y1"
            }],
            "chartOptions": {
              "mode": "COLOR"
            }
          }
        },
        "height": 4.0
      }]
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "displayName": "Ai",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:49:17.036234Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest",
    "dashboard": {
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "displayName": "Ai",
      "mosaicLayout": {
        "tiles": [{
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "groupByFields": ["resource.labels.method"]
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1"
              }]
            },
            "title": "Latency by API method (median)"
          },
          "width": 6.0,
          "height": 4.0
        }, {
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p50",
                "plotType": "LINE",
                "targetAxis": "Y1"
              }, {
                "legendTemplate": "p95",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                },
                "targetAxis": "Y1"
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE",
                "legendTemplate": "p99",
                "targetAxis": "Y1"
              }]
            }
          },
          "width": 6.0,
          "height": 4.0,
          "xPos": 6.0
        }, {
          "yPos": 4.0,
          "width": 6.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "groupByFields": ["resource.labels.method"],
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }]
            }
          },
          "height": 4.0
        }, {
          "height": 4.0,
          "yPos": 4.0,
          "xPos": 6.0,
          "width": 6.0,
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "legendTemplate": "p50",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }, {
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                      "perSeriesAligner": "ALIGN_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                },
                "legendTemplate": "p95",
                "targetAxis": "Y1",
                "plotType": "LINE"
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE",
                "targetAxis": "Y1",
                "legendTemplate": "p99"
              }],
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          }
        }, {
          "widget": {
            "xyChart": {
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    },
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    }
                  }
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Errors by API method"
          },
          "width": 6.0,
          "yPos": 8.0,
          "height": 4.0
        }, {
          "widget": {
            "title": "Traffic by response code",
            "xyChart": {
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "groupByFields": ["metric.labels.response_code"],
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "count/s"
                },
                "plotType": "LINE"
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "width": 6.0,
          "yPos": 8.0,
          "xPos": 6.0,
          "height": 4.0
        }],
        "columns": 12.0
      }
    }
  },
  "response": {
    "mosaicLayout": {
      "tiles": [{
        "height": 4.0,
        "widget": {
          "xyChart": {
            "dataSets": [{
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "groupByFields": ["resource.labels.method"],
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          },
          "title": "Latency by API method (median)"
        },
        "width": 6.0
      }, {
        "xPos": 6.0,
        "width": 6.0,
        "height": 4.0,
        "widget": {
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "legendTemplate": "p50",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              }
            }, {
              "targetAxis": "Y1",
              "plotType": "LINE",
              "legendTemplate": "p95",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              }
            }, {
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  }
                },
                "unitOverride": "second"
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }]
          },
          "title": "Overall latency"
        }
      }, {
        "yPos": 4.0,
        "height": 4.0,
        "width": 6.0,
        "widget": {
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "groupByFields": ["resource.labels.method"]
                  }
                },
                "unitOverride": "second"
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          },
          "title": "Latency by API method (median)"
        }
      }, {
        "widget": {
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "legendTemplate": "p50",
              "targetAxis": "Y1"
            }, {
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "plotType": "LINE",
              "targetAxis": "Y1",
              "legendTemplate": "p95"
            }, {
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "targetAxis": "Y1",
              "plotType": "LINE",
              "legendTemplate": "p99"
            }]
          },
          "title": "Overall latency"
        },
        "yPos": 4.0,
        "height": 4.0,
        "xPos": 6.0,
        "width": 6.0
      }, {
        "width": 6.0,
        "widget": {
          "xyChart": {
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  },
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                }
              }
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          },
          "title": "Errors by API method"
        },
        "height": 4.0,
        "yPos": 8.0
      }, {
        "height": 4.0,
        "yPos": 8.0,
        "xPos": 6.0,
        "width": 6.0,
        "widget": {
          "title": "Traffic by response code",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "groupByFields": ["metric.labels.response_code"],
                    "crossSeriesReducer": "REDUCE_SUM",
                    "perSeriesAligner": "ALIGN_RATE"
                  }
                },
                "unitOverride": "count/s"
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        }
      }],
      "columns": 12.0
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
    "displayName": "Ai"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:49:53.092443Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "dashboard": {
      "mosaicLayout": {
        "tiles": [{
          "height": 4.0,
          "width": 6.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "dataSets": [{
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "groupByFields": ["resource.labels.method"],
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  }
                },
                "plotType": "LINE"
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          }
        }, {
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p50",
                "targetAxis": "Y1"
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p95",
                "plotType": "LINE",
                "targetAxis": "Y1"
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "legendTemplate": "p99",
                "plotType": "LINE"
              }]
            },
            "title": "Overall latency"
          },
          "xPos": 6.0,
          "width": 6.0,
          "height": 4.0
        }, {
          "height": 4.0,
          "yPos": 4.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "groupByFields": ["resource.labels.method"],
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }]
            }
          },
          "width": 6.0
        }, {
          "width": 6.0,
          "yPos": 4.0,
          "xPos": 6.0,
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "legendTemplate": "p50",
                "plotType": "LINE",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                }
              }, {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "legendTemplate": "p95",
                "plotType": "LINE"
              }, {
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "legendTemplate": "p99"
              }]
            }
          },
          "height": 4.0
        }, {
          "height": 4.0,
          "widget": {
            "title": "Errors by API method",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    },
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    }
                  }
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }]
            }
          },
          "yPos": 8.0,
          "width": 6.0
        }, {
          "width": 6.0,
          "height": 4.0,
          "yPos": 8.0,
          "widget": {
            "title": "Traffic by response code",
            "xyChart": {
              "dataSets": [{
                "plotType": "LINE",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "unitOverride": "count/s",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_RATE",
                      "groupByFields": ["metric.labels.response_code"],
                      "crossSeriesReducer": "REDUCE_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                }
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "xPos": 6.0
        }, {
          "yPos": 12.0,
          "height": 4.0,
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "groupByFields": ["metric.labels.response_code_class"],
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                }
              }]
            },
            "title": "Latency by response code (median)"
          },
          "width": 6.0
        }],
        "columns": 12.0
      },
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "displayName": "Ai"
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest"
  },
  "response": {
    "displayName": "Ai",
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
    "mosaicLayout": {
      "tiles": [{
        "width": 6.0,
        "widget": {
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "groupByFields": ["resource.labels.method"],
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              }
            }]
          },
          "title": "Latency by API method (median)"
        },
        "height": 4.0
      }, {
        "height": 4.0,
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "dataSets": [{
              "legendTemplate": "p50",
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                },
                "unitOverride": "second"
              }
            }, {
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                  }
                }
              },
              "legendTemplate": "p95"
            }, {
              "plotType": "LINE",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "legendTemplate": "p99"
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          }
        },
        "xPos": 6.0,
        "width": 6.0
      }, {
        "width": 6.0,
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "groupByFields": ["resource.labels.method"],
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              }
            }]
          }
        },
        "yPos": 4.0,
        "height": 4.0
      }, {
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                },
                "unitOverride": "second"
              },
              "legendTemplate": "p50",
              "plotType": "LINE"
            }, {
              "plotType": "LINE",
              "legendTemplate": "p95",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                  }
                }
              },
              "targetAxis": "Y1"
            }, {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  }
                },
                "unitOverride": "second"
              },
              "legendTemplate": "p99",
              "plotType": "LINE",
              "targetAxis": "Y1"
            }]
          }
        },
        "width": 6.0,
        "height": 4.0,
        "yPos": 4.0,
        "xPos": 6.0
      }, {
        "widget": {
          "title": "Errors by API method",
          "xyChart": {
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  }
                }
              },
              "targetAxis": "Y1"
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          }
        },
        "width": 6.0,
        "height": 4.0,
        "yPos": 8.0
      }, {
        "height": 4.0,
        "widget": {
          "title": "Traffic by response code",
          "xyChart": {
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_SUM",
                    "perSeriesAligner": "ALIGN_RATE",
                    "groupByFields": ["metric.labels.response_code"]
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "count/s"
              }
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "xPos": 6.0,
        "yPos": 8.0,
        "width": 6.0
      }, {
        "widget": {
          "xyChart": {
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "groupByFields": ["metric.labels.response_code_class"],
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              }
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          },
          "title": "Latency by response code (median)"
        },
        "height": 4.0,
        "width": 6.0,
        "yPos": 12.0
      }],
      "columns": 12.0
    }
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:50:13.494399Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest",
    "dashboard": {
      "displayName": "Ai",
      "mosaicLayout": {
        "tiles": [{
          "width": 6.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "groupByFields": ["resource.labels.method"]
                    }
                  }
                }
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "height": 4.0
        }, {
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE",
                "legendTemplate": "p50",
                "targetAxis": "Y1"
              }, {
                "legendTemplate": "p95",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }, {
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p99",
                "targetAxis": "Y1"
              }],
              "chartOptions": {
                "mode": "COLOR"
              }
            },
            "title": "Overall latency"
          },
          "xPos": 6.0,
          "width": 6.0,
          "height": 4.0
        }, {
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "groupByFields": ["resource.labels.method"],
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE"
              }]
            },
            "title": "Latency by API method (median)"
          },
          "yPos": 4.0,
          "height": 4.0,
          "width": 6.0
        }, {
          "xPos": 6.0,
          "height": 4.0,
          "width": 6.0,
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "legendTemplate": "p50",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1"
              }, {
                "legendTemplate": "p95",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }, {
                "legendTemplate": "p99",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                      "perSeriesAligner": "ALIGN_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "targetAxis": "Y1"
              }],
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          },
          "yPos": 4.0
        }, {
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    },
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }]
            },
            "title": "Errors by API method"
          },
          "yPos": 8.0,
          "width": 6.0,
          "height": 4.0
        }, {
          "height": 4.0,
          "width": 6.0,
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "groupByFields": ["metric.labels.response_code"],
                      "crossSeriesReducer": "REDUCE_SUM",
                      "perSeriesAligner": "ALIGN_RATE"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "count/s"
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }],
              "chartOptions": {
                "mode": "COLOR"
              }
            },
            "title": "Traffic by response code"
          },
          "xPos": 6.0,
          "yPos": 8.0
        }, {
          "height": 4.0,
          "yPos": 12.0,
          "width": 6.0,
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "groupByFields": ["metric.labels.response_code_class"],
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Latency by response code (median)"
          }
        }, {
          "width": 6.0,
          "widget": {
            "title": "Errors by credential",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    },
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    }
                  }
                }
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "xPos": 6.0,
          "height": 4.0,
          "yPos": 12.0
        }],
        "columns": 12.0
      },
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd"
    }
  },
  "response": {
    "displayName": "Ai",
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "mosaicLayout": {
      "tiles": [{
        "widget": {
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "groupByFields": ["resource.labels.method"],
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              }
            }],
            "chartOptions": {
              "mode": "COLOR"
            }
          },
          "title": "Latency by API method (median)"
        },
        "height": 4.0,
        "width": 6.0
      }, {
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "legendTemplate": "p50",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "targetAxis": "Y1"
            }, {
              "legendTemplate": "p95",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "plotType": "LINE"
            }, {
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "targetAxis": "Y1",
              "legendTemplate": "p99"
            }],
            "chartOptions": {
              "mode": "COLOR"
            }
          }
        },
        "xPos": 6.0,
        "width": 6.0,
        "height": 4.0
      }, {
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "groupByFields": ["resource.labels.method"],
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "yPos": 4.0,
        "height": 4.0,
        "width": 6.0
      }, {
        "height": 4.0,
        "width": 6.0,
        "widget": {
          "title": "Overall latency",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  }
                }
              },
              "plotType": "LINE",
              "targetAxis": "Y1",
              "legendTemplate": "p50"
            }, {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "plotType": "LINE",
              "targetAxis": "Y1",
              "legendTemplate": "p95"
            }, {
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "yPos": 4.0,
        "xPos": 6.0
      }, {
        "width": 6.0,
        "widget": {
          "title": "Errors by API method",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  }
                }
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "yPos": 8.0,
        "height": 4.0
      }, {
        "xPos": 6.0,
        "yPos": 8.0,
        "width": 6.0,
        "height": 4.0,
        "widget": {
          "title": "Traffic by response code",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "unitOverride": "count/s",
                "timeSeriesFilter": {
                  "aggregation": {
                    "groupByFields": ["metric.labels.response_code"],
                    "perSeriesAligner": "ALIGN_RATE",
                    "crossSeriesReducer": "REDUCE_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        }
      }, {
        "width": 6.0,
        "widget": {
          "xyChart": {
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "groupByFields": ["metric.labels.response_code_class"],
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1"
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          },
          "title": "Latency by response code (median)"
        },
        "height": 4.0,
        "yPos": 12.0
      }, {
        "width": 6.0,
        "height": 4.0,
        "yPos": 12.0,
        "xPos": 6.0,
        "widget": {
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  },
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                }
              }
            }]
          },
          "title": "Errors by credential"
        }
      }],
      "columns": 12.0
    },
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {
  },
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:50:27.025569Z",
      "auth": {
      }
    },
    "destinationAttributes": {
    }
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [{
    "permission": "monitoring.dashboards.update",
    "granted": true,
    "resourceAttributes": {
      "service": "monitoring.googleapis.com",
      "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "type": "monitoring.googleapis.com/Dashboard"
    },
    "permissionType": "ADMIN_WRITE"
  }],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "dashboard": {
      "mosaicLayout": {
        "columns": 12.0,
        "tiles": [{
          "width": 6.0,
          "height": 4.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "groupByFields": ["resource.labels.method"],
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                }
              }]
            }
          }
        }, {
          "height": 4.0,
          "width": 6.0,
          "widget": {
            "title": "Overall latency",
            "xyChart": {
              "dataSets": [{
                "legendTemplate": "p50",
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "perSeriesAligner": "ALIGN_SUM"
                    }
                  }
                }
              }, {
                "legendTemplate": "p95",
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    }
                  }
                }
              }, {
                "targetAxis": "Y1",
                "legendTemplate": "p99",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "unitOverride": "second",
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    }
                  }
                }
              }],
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          },
          "xPos": 6.0
        }, {
          "width": 6.0,
          "height": 4.0,
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "dataSets": [{
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "groupByFields": ["resource.labels.method"],
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                }
              }],
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          },
          "yPos": 4.0
        }, {
          "xPos": 6.0,
          "width": 6.0,
          "height": 4.0,
          "widget": {
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "legendTemplate": "p50",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE"
              }, {
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    }
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p95"
              }, {
                "targetAxis": "Y1",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "legendTemplate": "p99"
              }]
            },
            "title": "Overall latency"
          },
          "yPos": 4.0
        }, {
          "yPos": 8.0,
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    },
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    }
                  }
                },
                "targetAxis": "Y1",
                "plotType": "LINE"
              }],
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Errors by API method"
          },
          "width": 6.0,
          "height": 4.0
        }, {
          "yPos": 8.0,
          "widget": {
            "xyChart": {
              "dataSets": [{
                "plotType": "LINE",
                "targetAxis": "Y1",
                "timeSeriesQuery": {
                  "unitOverride": "count/s",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_RATE",
                      "groupByFields": ["metric.labels.response_code"],
                      "crossSeriesReducer": "REDUCE_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                }
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Traffic by response code"
          },
          "xPos": 6.0,
          "height": 4.0,
          "width": 6.0
        }, {
          "widget": {
            "title": "Latency by response code (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                      "groupByFields": ["metric.labels.response_code_class"]
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "unitOverride": "second"
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }]
            }
          },
          "height": 4.0,
          "yPos": 12.0,
          "width": 6.0
        }, {
          "xPos": 6.0,
          "width": 6.0,
          "widget": {
            "title": "Errors by credential",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [{
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilterRatio": {
                    "numerator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                    },
                    "denominator": {
                      "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                    }
                  }
                },
                "targetAxis": "Y1"
              }],
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          },
          "yPos": 12.0,
          "height": 4.0
        }, {
          "width": 6.0,
          "height": 4.0,
          "widget": {
            "title": "Traffic by credential",
            "xyChart": {
              "dataSets": [{
                "timeSeriesQuery": {
                  "unitOverride": "count/s",
                  "timeSeriesFilter": {
                    "aggregation": {
                      "perSeriesAligner": "ALIGN_RATE",
                      "groupByFields": ["resource.labels.credential_id"],
                      "crossSeriesReducer": "REDUCE_SUM"
                    },
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                },
                "plotType": "LINE"
              }],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "yPos": 16.0
        }]
      },
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "displayName": "Ai"
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest"
  },
  "response": {
    "mosaicLayout": {
      "tiles": [{
        "height": 4.0,
        "widget": {
          "title": "Latency by API method (median)",
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM",
                    "groupByFields": ["resource.labels.method"]
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }]
          }
        },
        "width": 6.0
      }, {
        "widget": {
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              },
              "legendTemplate": "p50"
            }, {
              "targetAxis": "Y1",
              "legendTemplate": "p95",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              }
            }, {
              "legendTemplate": "p99",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }]
          },
          "title": "Overall latency"
        },
        "width": 6.0,
        "xPos": 6.0,
        "height": 4.0
      }, {
        "height": 4.0,
        "yPos": 4.0,
        "width": 6.0,
        "widget": {
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "groupByFields": ["resource.labels.method"],
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              },
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          },
          "title": "Latency by API method (median)"
        }
      }, {
        "height": 4.0,
        "yPos": 4.0,
        "widget": {
          "xyChart": {
            "xAxis": {
              "scale": "LINEAR"
            },
            "dataSets": [{
              "plotType": "LINE",
              "targetAxis": "Y1",
              "legendTemplate": "p50",
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "perSeriesAligner": "ALIGN_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                },
                "unitOverride": "second"
              }
            }, {
              "legendTemplate": "p95",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              },
              "targetAxis": "Y1"
            }, {
              "legendTemplate": "p99",
              "targetAxis": "Y1",
              "plotType": "LINE",
              "timeSeriesQuery": {
                "unitOverride": "second",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              }
            }],
            "chartOptions": {
              "mode": "COLOR"
            }
          },
          "title": "Overall latency"
        },
        "width": 6.0,
        "xPos": 6.0
      }, {
        "width": 6.0,
        "widget": {
          "title": "Errors by API method",
          "xyChart": {
            "dataSets": [{
              "plotType": "LINE",
              "targetAxis": "Y1",
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  },
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  }
                }
              }
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          }
        },
        "height": 4.0,
        "yPos": 8.0
      }, {
        "xPos": 6.0,
        "widget": {
          "title": "Traffic by response code",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "unitOverride": "count/s",
                "timeSeriesFilter": {
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_RATE",
                    "groupByFields": ["metric.labels.response_code"],
                    "crossSeriesReducer": "REDUCE_SUM"
                  },
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                }
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          }
        },
        "width": 6.0,
        "yPos": 8.0,
        "height": 4.0
      }, {
        "widget": {
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_latencies\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                    "groupByFields": ["metric.labels.response_code_class"],
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                },
                "unitOverride": "second"
              },
              "targetAxis": "Y1",
              "plotType": "LINE"
            }],
            "xAxis": {
              "scale": "LINEAR"
            },
            "chartOptions": {
              "mode": "COLOR"
            }
          },
          "title": "Latency by response code (median)"
        },
        "width": 6.0,
        "height": 4.0,
        "yPos": 12.0
      }, {
        "widget": {
          "title": "Errors by credential",
          "xyChart": {
            "dataSets": [{
              "plotType": "LINE",
              "timeSeriesQuery": {
                "timeSeriesFilterRatio": {
                  "denominator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\""
                  },
                  "numerator": {
                    "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\" AND (metric.labels.response_code_class\u003done_of(\"4xx\", \"5xx\"))"
                  }
                }
              },
              "targetAxis": "Y1"
            }],
            "chartOptions": {
              "mode": "COLOR"
            },
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "height": 4.0,
        "xPos": 6.0,
        "yPos": 12.0,
        "width": 6.0
      }, {
        "width": 6.0,
        "widget": {
          "title": "Traffic by credential",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [{
              "timeSeriesQuery": {
                "unitOverride": "count/s",
                "timeSeriesFilter": {
                  "filter": "resource.type\u003d\"consumed_api\" AND metric.type\u003d\"serviceruntime.googleapis.com/api/request_count\" AND project\u003d\"gen-lang-client-0049300763\" AND resource.labels.service\u003d\"servicehealth.googleapis.com\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_RATE",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": ["resource.labels.credential_id"]
                  }
                }
              },
              "plotType": "LINE",
              "targetAxis": "Y1"
            }],
            "xAxis": {
              "scale": "LINEAR"
            }
          }
        },
        "yPos": 16.0,
        "height": 4.0
      }],
      "columns": 12.0
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
    "displayName": "Ai"
  }
}{
  "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
  "status": {},
  "authenticationInfo": {
    "principalEmail": "chosen06021957@gmail.com"
  },
  "requestMetadata": {
    "callerIp": "2601:3cb:800:3000:c2c:a33c:9df5:6610",
    "callerSuppliedUserAgent": "Mozilla/5.0 (Linux; Android 16; SM-S176V Build/BP4A.251205.006; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/155.0.8059.43 Mobile Safari/537.36 Vespa/26.09.300800,gzip(gfe),gzip(gfe)",
    "requestAttributes": {
      "time": "2026-10-19T18:50:39.347878Z",
      "auth": {}
    },
    "destinationAttributes": {}
  },
  "serviceName": "monitoring.googleapis.com",
  "methodName": "google.monitoring.dashboard.v1.DashboardsService.UpdateDashboard",
  "authorizationInfo": [
    {
      "permission": "monitoring.dashboards.update",
      "granted": true,
      "resourceAttributes": {
        "service": "monitoring.googleapis.com",
        "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
        "type": "monitoring.googleapis.com/Dashboard"
      },
      "permissionType": "ADMIN_WRITE"
    }
  ],
  "resourceName": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
  "request": {
    "dashboard": {
      "name": "projects/gen-lang-client-0049300763/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd",
      "displayName": "Ai",
      "mosaicLayout": {
        "columns": 12.0,
        "tiles": [
          {
            "height": 4.0,
            "width": 6.0,
            "widget": {
              "xyChart": {
                "chartOptions": {
                  "mode": "COLOR"
                },
                "dataSets": [
                  {
                    "timeSeriesQuery": {
                      "unitOverride": "second",
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                          "groupByFields": [
                            "resource.labels.method"
                          ]
                        }
                      }
                    },
                    "targetAxis": "Y1",
                    "plotType": "LINE"
                  }
                ],
                "xAxis": {
                  "scale": "LINEAR"
                }
              },
              "title": "Latency by API method (median)"
            }
          },
          {
            "height": 4.0,
            "width": 6.0,
            "xPos": 6.0,
            "widget": {
              "xyChart": {
                "chartOptions": {
                  "mode": "COLOR"
                },
                "dataSets": [
                  {
                    "targetAxis": "Y1",
                    "plotType": "LINE",
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                        },
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                      },
                      "unitOverride": "second"
                    },
                    "legendTemplate": "p50"
                  },
                  {
                    "legendTemplate": "p95",
                    "plotType": "LINE",
                    "targetAxis": "Y1",
                    "timeSeriesQuery": {
                      "unitOverride": "second",
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                        }
                      }
                    }
                  },
                  {
                    "targetAxis": "Y1",
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                        },
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                      },
                      "unitOverride": "second"
                    },
                    "legendTemplate": "p99",
                    "plotType": "LINE"
                  }
                ],
                "xAxis": {
                  "scale": "LINEAR"
                }
              },
              "title": "Overall latency"
            }
          },
          {
            "width": 6.0,
            "yPos": 4.0,
            "height": 4.0,
            "widget": {
              "xyChart": {
                "xAxis": {
                  "scale": "LINEAR"
                },
                "dataSets": [
                  {
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "aggregation": {
                          "groupByFields": [
                            "resource.labels.method"
                          ],
                          "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                          "perSeriesAligner": "ALIGN_SUM"
                        },
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                      },
                      "unitOverride": "second"
                    },
                    "plotType": "LINE",
                    "targetAxis": "Y1"
                  }
                ],
                "chartOptions": {
                  "mode": "COLOR"
                }
              },
              "title": "Latency by API method (median)"
            }
          },
          {
            "width": 6.0,
            "yPos": 4.0,
            "xPos": 6.0,
            "height": 4.0,
            "widget": {
              "xyChart": {
                "dataSets": [
                  {
                    "legendTemplate": "p50",
                    "plotType": "LINE",
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                        }
                      },
                      "unitOverride": "second"
                    },
                    "targetAxis": "Y1"
                  },
                  {
                    "plotType": "LINE",
                    "legendTemplate": "p95",
                    "timeSeriesQuery": {
                      "unitOverride": "second",
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                        }
                      }
                    },
                    "targetAxis": "Y1"
                  },
                  {
                    "legendTemplate": "p99",
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_SUM",
                          "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                        }
                      },
                      "unitOverride": "second"
                    },
                    "plotType": "LINE",
                    "targetAxis": "Y1"
                  }
                ],
                "chartOptions": {
                  "mode": "COLOR"
                },
                "xAxis": {
                  "scale": "LINEAR"
                }
              },
              "title": "Overall latency"
            }
          },
          {
            "height": 4.0,
            "widget": {
              "title": "Errors by API method",
              "xyChart": {
                "dataSets": [
                  {
                    "targetAxis": "Y1",
                    "timeSeriesQuery": {
                      "timeSeriesFilterRatio": {
                        "numerator": {
                          "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com" AND (metric.labels.response_code_class=one_of("4xx", "5xx"))"
                        },
                        "denominator": {
                          "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                        }
                      }
                    },
                    "plotType": "LINE"
                  }
                ],
                "chartOptions": {
                  "mode": "COLOR"
                },
                "xAxis": {
                  "scale": "LINEAR"
                }
              }
            },
            "width": 6.0,
            "yPos": 8.0
          },
          {
            "yPos": 8.0,
            "height": 4.0,
            "widget": {
              "title": "Traffic by response code",
              "xyChart": {
                "chartOptions": {
                  "mode": "COLOR"
                },
                "dataSets": [
                  {
                    "plotType": "LINE",
                    "targetAxis": "Y1",
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "aggregation": {
                          "groupByFields": [
                            "metric.labels.response_code"
                          ],
                          "crossSeriesReducer": "REDUCE_SUM",
                          "perSeriesAligner": "ALIGN_RATE"
                        },
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                      },
                      "unitOverride": "count/s"
                    }
                  }
                ],
                "xAxis": {
                  "scale": "LINEAR"
                }
              }
            },
            "width": 6.0,
            "xPos": 6.0
          },
          {
            "yPos": 12.0,
            "height": 4.0,
            "width": 6.0,
            "widget": {
              "title": "Latency by response code (median)",
              "xyChart": {
                "dataSets": [
                  {
                    "timeSeriesQuery": {
                      "unitOverride": "second",
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                          "perSeriesAligner": "ALIGN_SUM",
                          "groupByFields": [
                            "metric.labels.response_code_class"
                          ]
                        }
                      }
                    },
                    "plotType": "LINE",
                    "targetAxis": "Y1"
                  }
                ],
                "chartOptions": {
                  "mode": "COLOR"
                },
                "xAxis": {
                  "scale": "LINEAR"
                }
              }
            }
          },
          {
            "yPos": 12.0,
            "height": 4.0,
            "widget": {
              "title": "Errors by credential",
              "xyChart": {
                "xAxis": {
                  "scale": "LINEAR"
                },
                "dataSets": [
                  {
                    "plotType": "LINE",
                    "targetAxis": "Y1",
                    "timeSeriesQuery": {
                      "timeSeriesFilterRatio": {
                        "denominator": {
                          "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                        },
                        "numerator": {
                          "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com" AND (metric.labels.response_code_class=one_of("4xx", "5xx"))"
                        }
                      }
                    }
                  }
                ],
                "chartOptions": {
                  "mode": "COLOR"
                }
              }
            },
            "width": 6.0,
            "xPos": 6.0
          },
          {
            "height": 4.0,
            "yPos": 16.0,
            "widget": {
              "title": "Traffic by credential",
              "xyChart": {
                "xAxis": {
                  "scale": "LINEAR"
                },
                "chartOptions": {
                  "mode": "COLOR"
                },
                "dataSets": [
                  {
                    "plotType": "LINE",
                    "timeSeriesQuery": {
                      "unitOverride": "count/s",
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "groupByFields": [
                            "resource.labels.credential_id"
                          ],
                          "perSeriesAligner": "ALIGN_RATE",
                          "crossSeriesReducer": "REDUCE_SUM"
                        }
                      }
                    },
                    "targetAxis": "Y1"
                  }
                ]
              }
            },
            "width": 6.0
          },
          {
            "yPos": 16.0,
            "height": 4.0,
            "widget": {
              "title": "Traffic by API method",
              "xyChart": {
                "dataSets": [
                  {
                    "timeSeriesQuery": {
                      "timeSeriesFilter": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                        "aggregation": {
                          "perSeriesAligner": "ALIGN_RATE",
                          "crossSeriesReducer": "REDUCE_SUM",
                          "groupByFields": [
                            "resource.labels.method"
                          ]
                        }
                      },
                      "unitOverride": "count/s"
                    },
                    "plotType": "LINE"
                  }
                ],
                "chartOptions": {
                  "mode": "COLOR"
                },
                "xAxis": {
                  "scale": "LINEAR"
                }
              }
            },
            "width": 6.0,
            "xPos": 6.0
          }
        ]
      }
    },
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.UpdateDashboardRequest"
  },
  "response": {
    "@type": "type.googleapis.com/google.monitoring.dashboard.v1.Dashboard",
    "displayName": "Ai",
    "mosaicLayout": {
      "tiles": [
        {
          "widget": {
            "title": "Latency by API method (median)",
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [
                {
                  "timeSeriesQuery": {
                    "unitOverride": "second",
                    "timeSeriesFilter": {
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                        "groupByFields": [
                          "resource.labels.method"
                        ],
                        "perSeriesAligner": "ALIGN_SUM"
                      },
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                    }
                  },
                  "plotType": "LINE",
                  "targetAxis": "Y1"
                }
              ],
              "xAxis": {
                "scale": "LINEAR"
              }
            }
          },
          "height": 4.0,
          "width": 6.0
        },
        {
          "xPos": 6.0,
          "width": 6.0,
          "height": 4.0,
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [
                {
                  "legendTemplate": "p50",
                  "targetAxis": "Y1",
                  "plotType": "LINE",
                  "timeSeriesQuery": {
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                        "perSeriesAligner": "ALIGN_SUM"
                      }
                    },
                    "unitOverride": "second"
                  }
                },
                {
                  "legendTemplate": "p95",
                  "targetAxis": "Y1",
                  "plotType": "LINE",
                  "timeSeriesQuery": {
                    "unitOverride": "second",
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                        "perSeriesAligner": "ALIGN_SUM"
                      }
                    }
                  }
                },
                {
                  "legendTemplate": "p99",
                  "targetAxis": "Y1",
                  "timeSeriesQuery": {
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                        "perSeriesAligner": "ALIGN_SUM"
                      }
                    },
                    "unitOverride": "second"
                  },
                  "plotType": "LINE"
                }
              ]
            },
            "title": "Overall latency"
          }
        },
        {
          "height": 4.0,
          "widget": {
            "xyChart": {
              "dataSets": [
                {
                  "timeSeriesQuery": {
                    "timeSeriesFilter": {
                      "aggregation": {
                        "groupByFields": [
                          "resource.labels.method"
                        ],
                        "perSeriesAligner": "ALIGN_SUM",
                        "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                      },
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                    },
                    "unitOverride": "second"
                  },
                  "targetAxis": "Y1",
                  "plotType": "LINE"
                }
              ],
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              }
            },
            "title": "Latency by API method (median)"
          },
          "width": 6.0,
          "yPos": 4.0
        },
        {
          "width": 6.0,
          "widget": {
            "xyChart": {
              "dataSets": [
                {
                  "timeSeriesQuery": {
                    "timeSeriesFilter": {
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                        "perSeriesAligner": "ALIGN_SUM"
                      },
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                    },
                    "unitOverride": "second"
                  },
                  "plotType": "LINE",
                  "targetAxis": "Y1",
                  "legendTemplate": "p50"
                },
                {
                  "targetAxis": "Y1",
                  "timeSeriesQuery": {
                    "unitOverride": "second",
                    "timeSeriesFilter": {
                      "aggregation": {
                        "perSeriesAligner": "ALIGN_SUM",
                        "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                      },
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                    }
                  },
                  "legendTemplate": "p95",
                  "plotType": "LINE"
                },
                {
                  "timeSeriesQuery": {
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_PERCENTILE_99",
                        "perSeriesAligner": "ALIGN_SUM"
                      }
                    },
                    "unitOverride": "second"
                  },
                  "targetAxis": "Y1",
                  "plotType": "LINE",
                  "legendTemplate": "p99"
                }
              ],
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              }
            },
            "title": "Overall latency"
          },
          "xPos": 6.0,
          "height": 4.0,
          "yPos": 4.0
        },
        {
          "yPos": 8.0,
          "width": 6.0,
          "widget": {
            "xyChart": {
              "dataSets": [
                {
                  "timeSeriesQuery": {
                    "timeSeriesFilterRatio": {
                      "denominator": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                      },
                      "numerator": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com" AND (metric.labels.response_code_class=one_of("4xx", "5xx"))"
                      }
                    }
                  },
                  "plotType": "LINE",
                  "targetAxis": "Y1"
                }
              ],
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Errors by API method"
          },
          "height": 4.0
        },
        {
          "widget": {
            "title": "Traffic by response code",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [
                {
                  "timeSeriesQuery": {
                    "unitOverride": "count/s",
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_SUM",
                        "groupByFields": [
                          "metric.labels.response_code"
                        ],
                        "perSeriesAligner": "ALIGN_RATE"
                      }
                    }
                  },
                  "plotType": "LINE",
                  "targetAxis": "Y1"
                }
              ],
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          },
          "xPos": 6.0,
          "height": 4.0,
          "width": 6.0,
          "yPos": 8.0
        },
        {
          "width": 6.0,
          "widget": {
            "title": "Latency by response code (median)",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [
                {
                  "plotType": "LINE",
                  "timeSeriesQuery": {
                    "unitOverride": "second",
                    "timeSeriesFilter": {
                      "aggregation": {
                        "perSeriesAligner": "ALIGN_SUM",
                        "crossSeriesReducer": "REDUCE_PERCENTILE_50",
                        "groupByFields": [
                          "metric.labels.response_code_class"
                        ]
                      },
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_latencies" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                    }
                  },
                  "targetAxis": "Y1"
                }
              ],
              "chartOptions": {
                "mode": "COLOR"
              }
            }
          },
          "yPos": 12.0,
          "height": 4.0
        },
        {
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "xAxis": {
                "scale": "LINEAR"
              },
              "dataSets": [
                {
                  "targetAxis": "Y1",
                  "plotType": "LINE",
                  "timeSeriesQuery": {
                    "timeSeriesFilterRatio": {
                      "numerator": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com" AND (metric.labels.response_code_class=one_of("4xx", "5xx"))"
                      },
                      "denominator": {
                        "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com""
                      }
                    }
                  }
                }
              ]
            },
            "title": "Errors by credential"
          },
          "height": 4.0,
          "xPos": 6.0,
          "yPos": 12.0,
          "width": 6.0
        },
        {
          "height": 4.0,
          "widget": {
            "xyChart": {
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [
                {
                  "targetAxis": "Y1",
                  "timeSeriesQuery": {
                    "unitOverride": "count/s",
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "perSeriesAligner": "ALIGN_RATE",
                        "groupByFields": [
                          "resource.labels.credential_id"
                        ],
                        "crossSeriesReducer": "REDUCE_SUM"
                      }
                    }
                  },
                  "plotType": "LINE"
                }
              ],
              "xAxis": {
                "scale": "LINEAR"
              }
            },
            "title": "Traffic by credential"
          },
          "yPos": 16.0,
          "width": 6.0
        },
        {
          "xPos": 6.0,
          "yPos": 16.0,
          "widget": {
            "title": "Traffic by API method",
            "xyChart": {
              "xAxis": {
                "scale": "LINEAR"
              },
              "chartOptions": {
                "mode": "COLOR"
              },
              "dataSets": [
                {
                  "timeSeriesQuery": {
                    "unitOverride": "count/s",
                    "timeSeriesFilter": {
                      "filter": "resource.type="consumed_api" AND metric.type="serviceruntime.googleapis.com/api/request_count" AND project="gen-lang-client-0049300763" AND resource.labels.service="servicehealth.googleapis.com"",
                      "aggregation": {
                        "crossSeriesReducer": "REDUCE_SUM",
                        "perSeriesAligner": "ALIGN_RATE",
                        "groupByFields": [
                          "resource.labels.method"
                        ]
                      }
                    }
                  },
                  "targetAxis": "Y1",
                  "plotType": "LINE"
                }
              ]
            }
          },
          "width": 6.0,
          "height": 4.0
        }
      ],
      "columns": 12.0
    },
    "name": "projects/992786648543/dashboards/d310a792-322e-45a5-8d74-ccba6e33b2cd"
  }
}
