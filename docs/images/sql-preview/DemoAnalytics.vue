<template>
  <div class="analytics data-view">
    <div class="query-header">
      <ButtonGroup>
        <Button label="Builder" severity="secondary" />
        <Button label="SQL" />
      </ButtonGroup>
      <strong>Inference volume · local and cloud models</strong>
      <span class="sample-label">Sample data</span>
    </div>
    <div class="demo-notice" role="note">
      Illustrative sample data · 48 hours · 3 models. Query results computed locally with DuckDB.
      This screenshot preview is not a production deployment.
    </div>
    <SqlExplorer :initial-sql="query" demo-preview />
    <QueryResult compact :result="result" :loading="false" :error="null" :initial-view="{
      vizType: 'bar', stacked: true, xCol: 'hour', yCol: 'inferences', seriesCol: 'model_name',
      display: {}, statValueColumn: null, statReducer: 'last',
    }" />
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import Button from 'primevue/button'
import ButtonGroup from 'primevue/buttongroup'
import SqlExplorer from '@/components/SqlExplorer.vue'
import QueryResult from '@/components/QueryResult.vue'
import type { QueryResultData } from '@/services/analyticsService'
import query from '@/services/demo/screenshot-query.sql?raw'
import fixture from '@/services/demo/screenshot-result.json'
const result = ref<QueryResultData | null>(null)
onMounted(async () => { await nextTick(); result.value = fixture })
</script>
<style scoped>
.analytics { display: flex; flex-direction: column; gap: 1rem; }
.query-header { display: flex; align-items: center; flex-wrap: wrap; gap: 1rem; }
.sample-label { color: var(--text-color-secondary); margin-left: auto; }
.demo-notice { padding: .75rem 1rem; border: 1px solid var(--surface-border); border-radius: 8px;
  color: var(--text-color-secondary); line-height: 1.5; }
</style>
