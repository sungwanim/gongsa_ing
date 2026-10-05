import { CLASSES, DEFECT_ORDER } from '../config'

export default function DefectGlossary() {
  return (
    <>
      <div className="glossary">
        {DEFECT_ORDER.map((k) => (
          <div className="card gloss" key={k}>
            <span className="gloss-dot" style={{ background: CLASSES[k].color }} />
            <h3>
              {CLASSES[k].label} <small>{CLASSES[k].en}</small>
            </h3>
            <p>{CLASSES[k].desc}</p>
          </div>
        ))}
      </div>
      <p className="hint">설명은 샘플 이미지를 관찰해 정리한 초안이에요. 논문의 공식 정의와는 다를 수 있어요.</p>
    </>
  )
}
