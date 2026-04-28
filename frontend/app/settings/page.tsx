'use client'
import { useEffect, useState } from 'react'
import { Save, CheckCircle } from 'lucide-react'
import { Topbar } from '@/components/layout/Topbar'
import { ApiConfigPanel } from '@/components/settings/ApiConfig'
import { ModelParams } from '@/components/settings/ModelParams'
import { PipelineConfigPanel } from '@/components/settings/PipelineConfig'
import { ExpertConfigPanel } from '@/components/settings/ExpertConfig'
import { useStore } from '@/lib/store'
import { loadConfig, saveConfig } from '@/lib/api'
import type { ApiConfig, PipelineConfig, AnnotationExpertConfig } from '@/lib/types'

export default function SettingsPage() {
  const { config, setConfig } = useStore()
  const [saved, setSaved] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    setFetchError(null)
    loadConfig().then((raw) => {
      const r = raw as Record<string, unknown>
      setConfig({
        apiProvider:    String(r.api_provider  ?? ''),
        apiKey:         String(r.api_key       ?? ''),
        apiModel:       String(r.api_model     ?? ''),
        apiEndpoint:    String(r.api_endpoint  ?? ''),
        apiVersion:     String(r.api_version   ?? ''),
        temperature:    Number(r.temperature   ?? 0.7),
        topP:           Number(r.top_p         ?? 0.95),
        visualFactor:   Number(r.visual_factor ?? 0.5),
        dataPath:       String(r.data_path     ?? ''),
        sampleId:       String(r.sample_id     ?? ''),
        nClusters:      Number(r.n_clusters    ?? 7),
        methods:        Array.isArray(r.methods) && (r.methods as string[]).length > 0 ? r.methods as string[] : ['IRIS', 'BASS', 'DR-SC', 'BayesSpace', 'SEDR', 'GraphST', 'STAGATE', 'stLearn'],
        skipToolRunner: Boolean(r.skip_tool_runner ?? false),
        skipScoring:    Boolean(r.skip_scoring     ?? false),
        annotationWMarker:       Number(r.annotation_w_marker       ?? 0.30),
        annotationWPathway:      Number(r.annotation_w_pathway      ?? 0.20),
        annotationWSpatial:      Number(r.annotation_w_spatial      ?? 0.30),
        annotationWVlm:          Number(r.annotation_w_vlm          ?? 0.20),
        annotationMaxRounds:     Number(r.annotation_max_rounds     ?? 3),
        annotationStandardScore: Number(r.annotation_standard_score ?? 0.65),
        annotationVlmRequired:   Boolean(r.annotation_vlm_required  ?? true),
        annotationVlmMinScore:   Number(r.annotation_vlm_min_score  ?? 0.30),
      })
    }).catch((err) => {
      setFetchError(`Failed to load config from API: ${err.message}. Is the FastAPI server running on localhost:8000?`)
    })
  }, [setConfig])

  const handleApiChange = (key: keyof ApiConfig, value: string) =>
    setConfig({ [key]: value } as Partial<ApiConfig>)

  const handleParamChange = (key: string, value: number) =>
    setConfig({ [key]: value } as Partial<ApiConfig>)

  const handlePipelineChange = (key: keyof PipelineConfig, value: unknown) =>
    setConfig({ [key]: value } as Partial<PipelineConfig>)

  const handleExpertChange = (key: keyof AnnotationExpertConfig, value: number | boolean) =>
    setConfig({ [key]: value } as Partial<AnnotationExpertConfig>)

  const handleSave = async () => {
    await saveConfig({
      api_provider:     config.apiProvider,
      api_key:          config.apiKey,
      api_model:        config.apiModel,
      api_endpoint:     config.apiEndpoint,
      api_version:      config.apiVersion,
      temperature:      config.temperature,
      top_p:            config.topP,
      visual_factor:    config.visualFactor,
      data_path:        config.dataPath,
      sample_id:        config.sampleId,
      n_clusters:       config.nClusters,
      methods:          config.methods,
      skip_tool_runner:          config.skipToolRunner,
      skip_scoring:              config.skipScoring,
      annotation_w_marker:       config.annotationWMarker,
      annotation_w_pathway:      config.annotationWPathway,
      annotation_w_spatial:      config.annotationWSpatial,
      annotation_w_vlm:          config.annotationWVlm,
      annotation_max_rounds:     config.annotationMaxRounds,
      annotation_standard_score: config.annotationStandardScore,
      annotation_vlm_required:   config.annotationVlmRequired,
      annotation_vlm_min_score:  config.annotationVlmMinScore,
    }).catch(console.error)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <Topbar title="Settings" />
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
        {fetchError && (
          <div style={{
            background: 'rgba(239,68,68,0.06)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: '8px',
            padding: '10px 14px',
            marginBottom: '16px',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--err)',
            lineHeight: 1.5,
          }}>
            {fetchError}
          </div>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '32px', maxWidth: '1300px' }}>
          {/* Left column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <section>
              <div className="section-head">API Configuration</div>
              <ApiConfigPanel config={config} onChange={handleApiChange} />
            </section>

            <div style={{ height: '1px', background: 'var(--rule)' }} />

            <section>
              <div className="section-head">Model Parameters</div>
              <ModelParams
                temperature={config.temperature ?? 0.7}
                topP={config.topP ?? 0.95}
                visualFactor={config.visualFactor ?? 0.5}
                onChange={handleParamChange}
              />
            </section>
          </div>

          {/* Middle column */}
          <section>
            <div className="section-head">Pipeline Configuration</div>
            <PipelineConfigPanel config={config} onChange={handlePipelineChange} />
          </section>

          {/* Right column */}
          <section>
            <div className="section-head">Annotation Experts</div>
            <ExpertConfigPanel config={config} onChange={handleExpertChange} />
          </section>
        </div>

        {/* Save row */}
        <div style={{ marginTop: '28px', display: 'flex', alignItems: 'center', gap: '12px', paddingBottom: '8px' }}>
          <button className="btn-primary" onClick={handleSave}>
            <Save size={13} strokeWidth={2} />
            Save Settings
          </button>
          {saved && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px', animation: 'fade-in 0.2s ease-out' }}>
              <CheckCircle size={13} style={{ color: 'var(--ok)' }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ok)' }}>
                Saved
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
